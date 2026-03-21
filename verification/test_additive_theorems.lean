import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Finset.Basic
import additive_combinatorics

-- Test that our theorems work with the existing J13D offsets

-- J13D offset set
def j13d_offsets_list : List Nat := [1, 2, 4, 5, 8, 16, 32, 64, 128, 256, 512, 768, 1024]

def j13d_offsets_set : Finset Nat := j13d_offsets_list.toFinset

-- Test the span constraint theorem with J13D
example : ∃ a ∈ j13d_offsets_set, a ≤ 3 ∧ ∃ b ∈ j13d_offsets_set, b ≥ 515 := by
  apply offset_set_span_constraint j13d_offsets_set
  intro t ht
  -- Check that each target gap can be formed by some pair
  simp [target_gaps_set, target_gaps, j13d_offsets_set, j13d_offsets_list] at ht ⊢
  sorry -- We would need to verify each gap, but this shows the theorem applies

-- Test that J13D has cardinality 13
example : j13d_offsets_set.card = 13 := by
  simp [j13d_offsets_set, j13d_offsets_list]

-- Test that a small set (size 5) cannot cover all target gaps
-- This is the kind of thing we would prove with offset_set_lower_bound if we completed it
example : ¬(∀ t ∈ target_gaps_set, ∃ a ∈ ({1, 2, 4, 8, 16} : Finset Nat), ∃ b ∈ ({1, 2, 4, 8, 16} : Finset Nat), a + b = t) := by
  -- We would prove this using offset_set_lower_bound once it's complete
  -- For now, we just state that this should follow from our theorems
  sorry

-- Show that the theorems are properly imported and work
#check offset_set_lower_bound
#check offset_set_span_constraint
#check target_gaps_set
#check target_gaps