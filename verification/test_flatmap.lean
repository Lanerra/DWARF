import Mathlib.Data.List.Basic

example : [1, 2].flatMap (fun x => [x + 1]) = [2, 3] := by rfl
