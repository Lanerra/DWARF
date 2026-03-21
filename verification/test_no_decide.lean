import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic

def N : Nat := 2048

def j12_offsets : List Nat := [1, 2, 4, 8, 16, 64, 96, 192, 384, 512, 768, 1024]
def j13d_offsets : List Nat := [1, 2, 4, 5, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
def passkey_distances : List Nat := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
def passkey_gaps : List Nat := passkey_distances.map (· + 5)

example : 7 ∈ passkey_gaps := by decide
