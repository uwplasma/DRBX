"""Addressable spawned lanes with explicit shutdown of their tracer pools."""

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
import os
import signal
from time import monotonic
from traceback import format_exc


def _serve_plane(commands, results, closed, initialize, initargs):
    tracers = None
    try:
        initialization_error = None
        try:
            # The coordinator can reap this lane and nested children together
            # if a broken worker cannot finish a graceful close.
            os.setsid()
            tracers = initialize(*initargs)
        except Exception:
            initialization_error = format_exc()
        while (message := commands.recv()) is not None:
            work, arguments = message
            error, value = initialization_error, None
            if error is None:
                try:
                    value = work(*arguments)
                except Exception:
                    error = format_exc()
            results.send((error, value))
    except (EOFError, BrokenPipeError):
        # The coordinator can close the result pipe to discard in-flight work.
        pass
    finally:
        try:
            if tracers is not None:
                tracers.shutdown(wait=True, cancel_futures=True)
        except BaseException:
            closed.send_bytes(b'failed')
            raise
        else:
            closed.send_bytes(b'closed')


@dataclass
class _PlaneLane:
    process: mp.Process
    commands: Connection
    results: Connection
    closed: Connection


def _start_lane(context, initialize, initargs):
    commands, send_command = context.Pipe(duplex=False)
    receive_result, results = context.Pipe(duplex=False)
    receive_closed, closed = context.Pipe(duplex=False)
    process = context.Process(target=_serve_plane, args=(commands, results, closed, initialize, initargs),
                              name='polygon-plane')
    try:
        process.start()
    except BaseException:
        for pipe in (send_command, receive_result, receive_closed):
            pipe.close()
        raise
    finally:
        for pipe in (commands, results, closed):
            pipe.close()
    return _PlaneLane(process, send_command, receive_result, receive_closed)


def _kill_lane(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        # The child may have failed before creating its own group.
        if process.is_alive():
            process.kill()


class PlanePool:
    def __init__(self, lanes, initialize, initargs=(), *, shutdown_timeout=15.):
        self._lanes = []
        self._shutdown_timeout = shutdown_timeout
        self._closed = False
        try:
            context = mp.get_context('spawn')
            for _ in range(lanes):
                self._lanes.append(_start_lane(context, initialize, initargs))
        except BaseException as error:
            self._close_after_error(error)
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, traceback):
        if error is None:
            self.shutdown()
        else:
            self._close_after_error(error)

    def _close_after_error(self, error):
        try:
            self.shutdown()
        except RuntimeError as cleanup:
            # The phase failure remains the primary error.
            error.add_note(str(cleanup))

    def map(self, work, *iterables):
        """Keep one active unit per lane and return results in submission order."""
        pending = iter(enumerate(zip(*iterables, strict=True)))
        active, completed = {}, {}

        def dispatch(lane):
            job = next(pending, None)
            if job is not None:
                index, arguments = job
                lane.commands.send((work, arguments))
                active[lane.results] = lane, index

        try:
            for lane in self._lanes:
                dispatch(lane)
            while active:
                for connection in wait(active):
                    lane, index = active.pop(connection)
                    try:
                        error, value = connection.recv()
                    except EOFError as lost:
                        raise RuntimeError(f'Polygon plane {lane.process.pid} exited without a result') from lost
                    if error is not None:
                        raise RuntimeError(f'Polygon plane {lane.process.pid} failed:\n{error}')
                    completed[index] = value
                    dispatch(lane)
            return [completed[index] for index in range(len(completed))]
        except BaseException as error:
            # End every private writer before the calling phase drops its files.
            self._close_after_error(error)
            raise

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        deadline = monotonic()+self._shutdown_timeout
        problems = {}
        for lane in self._lanes:
            try:
                lane.commands.send(None)
            except (OSError, EOFError):
                pass  # The acknowledgment below diagnoses a lost plane.
            finally:
                lane.commands.close()
                lane.results.close()

        for lane in self._lanes:
            process = lane.process
            try:
                if not lane.closed.poll(max(0., deadline-monotonic())):
                    problems[process.pid] = 'no tracer shutdown acknowledgment before deadline'
                elif lane.closed.recv_bytes(16) != b'closed':
                    problems[process.pid] = 'tracer shutdown raised in the plane'
                else:
                    process.join(max(0., deadline-monotonic()))
                    if process.is_alive():
                        problems[process.pid] = 'plane did not exit before deadline'
                    elif process.exitcode != 0:
                        problems[process.pid] = f'plane exited with code {process.exitcode}'
            except (EOFError, OSError):
                problems[process.pid] = 'plane exited without confirming tracer shutdown'
            finally:
                lane.closed.close()

        if problems:
            for lane in self._lanes:
                process = lane.process
                if process.pid in problems:
                    try:
                        _kill_lane(process)
                    except OSError as error:
                        problems[process.pid] += f', forced cleanup failed: {error}'
            deadline = monotonic()+5.
            for lane in self._lanes:
                process = lane.process
                if process.pid in problems:
                    process.join(max(0., deadline-monotonic()))
                    if process.is_alive():
                        problems[process.pid] += ', plane is still alive'
        for lane in self._lanes:
            if not lane.process.is_alive():
                lane.process.close()
        if problems:
            detail = '; '.join(f'PID {pid}: {reason}' for pid, reason in problems.items())
            raise RuntimeError(f'Polygon plane shutdown failed (limit {self._shutdown_timeout:g} s): {detail}')
