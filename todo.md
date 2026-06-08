DONE - FIX CLONE NODE (add contiguity for cloning views)
DONE - Write odometer tensor printing algo (with ellipsis ...)
- Fix ownership bug where clearing m_op of one node doesnt clear of others. Make m_op SHARED like storage.
- Apply for condition where ellipsis is printed only if dim > 2 * edge_items
- Apply for scalars in every scenario (broadcasts, etc)
- Test out everything where you can print data and shapes (.realize())
- Write reductions (Sum, Mean)
- Start autograd stuff
- When needed split the project into files so you DRY in realize() and backward()