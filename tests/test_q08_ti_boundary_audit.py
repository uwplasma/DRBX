"""Reporting tests; the diagnostic never changes a frozen acceptance gate."""
import unittest
from types import SimpleNamespace
import numpy as np
from scripts.q08_ti_boundary_audit import difference_summary, query_context


class BoundaryAuditTests(unittest.TestCase):
    def test_repeated_failure_fraction_is_reported_with_values_and_index(self):
        expected=np.zeros((1,3,2));actual=expected.copy()
        actual[0,1,1]=1.14892578125*32*np.finfo(float).eps
        record=difference_summary(actual,expected,np.array([1]),32)
        self.assertEqual(record['max_budget_fraction'],1.14892578125)
        self.assertEqual(record['over_budget'],1)
        self.assertEqual(record['worst']['index'],[0,1,1])
        self.assertEqual(float.fromhex(record['worst']['actual_hex']),actual[0,1,1])
        self.assertTrue(record['nonwall_exact'])

    def test_padding_error_is_not_hidden_by_small_roundoff(self):
        expected=np.zeros((1,3,2));actual=expected.copy();actual[0,0,0]=1e-17
        record=difference_summary(actual,expected,np.array([1]),32)
        self.assertEqual(record['over_budget'],0)
        self.assertFalse(record['nonwall_exact'])

    def test_portable_fixture_policy_reports_its_actual_budget(self):
        expected=np.array([[[-.8559914628299974]]]);actual=np.array([[[-.8559914628299645]]])
        record=difference_summary(actual,expected,np.array([0]),atol=1e-12,rtol=1e-13)
        self.assertEqual(record['over_budget'],0)
        self.assertAlmostEqual(record['worst']['budget'],1e-12+1e-13*abs(expected.item()),places=25)

    def test_nonfinite_and_shape_errors_fail(self):
        expected=np.zeros((1,3,2))
        for actual in (expected.astype(np.float32),expected[0],np.full_like(expected,np.nan)):
            with self.assertRaises(ValueError):difference_summary(actual,expected,np.array([1]),32)

    def test_query_context_exposes_cancelling_normal_terms(self):
        bank=SimpleNamespace(raw=np.array([8,9]),wall_index=np.array([1]),
            wall_node_query=np.array([[0]]),query_table=np.array([[1.,2.,3.]]),
            boundary_wall_normal=np.array([[[100.,-100.,1.]]]))
        gradient=np.ones((1,1,5,3))
        common=SimpleNamespace(fields=lambda p:(None,gradient))
        result=query_context(bank,0,3,[0,1,0],common,lambda p,case,c:(None,gradient[:,0]))
        self.assertEqual(result['raw_id'],9)
        self.assertEqual(result['original_normal_products'],[100.,-100.,1.])
        self.assertEqual(result['original_sum_abs_products'],201.)


if __name__=='__main__':unittest.main()
