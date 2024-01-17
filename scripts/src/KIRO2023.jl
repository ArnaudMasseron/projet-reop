using JSON

include("utils.jl")
include("instance.jl")
include("solution.jl")
include("parsing.jl")
include("eval.jl")

export Instance, Solution
export read_instance, read_solution, write_solution
export construction_cost, operational_cost, cost
export is_feasible


tiny = read_instance("C:/Users/masse/Documents/Ecole/ENPC/2A/REOP/Projet/sujet_projet_REOP_2324/KIRO-tiny_20.json")
tiny_sol = read_solution("C:/Users/masse/Documents/Ecole/ENPC/2A/REOP/Projet/sujet_projet_REOP_2324/KIRO-tiny_sol_20.json")
#cost(tiny_sol, tiny)

#sol = read_solution("C:/Users/masse/Documents/Ecole/ENPC/2A/REOP/Projet/sujet_projet_REOP_2324/sol.json")
#cost(sol, tiny)