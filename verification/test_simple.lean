import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic

def N : Nat := 2048

def reachableInHops (offsets : List Nat) (d : Nat) (maxHops : Nat) : Bool :=
  let rec go (frontier : List Nat) (visited : List Nat) (hopsLeft : Nat) : Bool :=
    if hopsLeft == 0 then frontier.contains d
    else
      let next := frontier.flatMap (fun x => offsets.map (fun o => x + o))
        |>.filter (fun x => x <= d && !visited.contains x)
        |>.eraseDups
      if next.contains d then true
      else go next (visited ++ next) (hopsLeft - 1)
  go [0] [0] maxHops

example : reachableInHops [1, 2, 4] 7 2 = false := by
  unfold reachableInHops
  decide
