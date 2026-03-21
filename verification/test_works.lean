import Mathlib.Data.Nat.Basic

def N : Nat := 2048

def j12_offsets : List Nat := [1, 2, 4, 8, 16, 64, 96, 192, 384, 512, 768, 1024]

example : 7 ∉ j12_offsets := by
  decide
