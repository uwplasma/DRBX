# P06 actual-HSX boundary-functional fixture

This bounded fixture contains one N32 radial-wall cell from the parent P06
selection. It retains the real HSX owner-observation matrix, frozen donors and
weights, q3 wall polynomial value/gradient rows, contravariant metric,
Jacobian, logical quadrature weights, and dynamic owner values.

`generate_fixture.py` records the parent artifact and donor hash in the JSON
sidecar. The package geometry/data separation is tested explicitly: metric and
polynomial rows prepare the fixed maps, while boundary values are supplied only
at application time. The fixture is a regression and transformation test, not
a global accuracy or convergence claim.
