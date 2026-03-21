import Mathlib.Data.List.Basic

example (xs : List Nat) (f : Nat -> List Nat) : List Nat := xs.bind f
example (xs : List Nat) (f : Nat -> List Nat) : List Nat := xs.flatMap f
