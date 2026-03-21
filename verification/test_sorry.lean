noncomputable def test : Nat := 
  let rec go (n : Nat) : Nat := 
    if n = 0 then 0 else go (n - 1)
  go 10

#check test
