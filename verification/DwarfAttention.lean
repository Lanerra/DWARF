import DwarfAttention

-- Test script to find all 2-hop pairs for each gap
example : findTwoHopPairs J13D 13 = [(5, 8)] := by native_decide
example : findTwoHopPairs J13D 21 = [(5, 16)] := by native_decide
example : findTwoHopPairs J13D 261 = [(5, 256)] := by native_decide
example : findTwoHopPairs J13D 517 = [(5, 512)] := by native_decide
example : findTwoHopPairs J13D 1029 = [(5, 1024)] := by native_decide

-- Check with ablated set
example : findTwoHopPairs J13D_ablated 13 = [] := by native_decide
example : findTwoHopPairs J13D_ablated 21 = [] := by native_decide
example : findTwoHopPairs J13D_ablated 261 = [] := by native_decide
example : findTwoHopPairs J13D_ablated 517 = [] := by native_decide
example : findTwoHopPairs J13D_ablated 1029 = [] := by native_decide
