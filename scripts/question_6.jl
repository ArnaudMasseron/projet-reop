"""
    Remarques

La fonction objectif semble etre codee correctement: j'ai compare les valeurs qu'on obtenait avec
ce programme et le programme d'evaluation des solutions sur le github de leo baty et a chaque fois
j'obtient la meme solution avec les 2 programmes. Cependant Les solutions obtenues avec ce programme 
pour l'instance tiny sont bizarres: il n'y a aucune station qui est construite.

Le fonctions linearize... semblent etre codees correctement: je les ai testees dans des problemes
d'optimisation simples et elles fonctionnaient bien.

Le fichier tiny_sol.json ne semble pas etre la solution optimale pour l'instance tiny.json car en modifiant
un peu la solution codee dans ce fichier on peut facilement faire baisser le cout.

Il faut encore coder la fonction qui permet d'ecrire la solution dans un fichier JSON.

Le programme est tres lent avec l'instance small: je l'ai fait tourner pendant une heure et il ca ne m'avait
pas fourni de solution.
"""


using JSON, JuMP, Gurobi
import LinearAlgebra: norm

### Chargement des donnees
file_path = joinpath(@__DIR__, "instances/KIRO-small.json")
data = JSON.parsefile(file_path)

Vˢ = data["substation_locations"]
Vᵗ = data["wind_turbines"]
S = data["substation_types"]
Q⁰ = data["land_substation_cable_types"]
Qˢ = data["substation_substation_cable_types"]
Ω = data["wind_scenarios"]

E⁰ = [(0, v) for v ∈ 1:length(Vˢ)]
Eˢ = []
for v ∈ 1:length(Vˢ)
    for v_prime ∈ v+1:length(Vˢ)
        push!(Eˢ, (v, v_prime))
    end
end

indices_Vˢ = 1:length(Vˢ)
indices_Vᵗ = 1:length(Vᵗ)
indices_S = 1:length(S)
indices_E⁰ = indices_Vˢ
indices_Eˢ = 1:length(Eˢ)
indices_Q⁰ = 1:length(Q⁰)
indices_Qˢ = 1:length(Qˢ)
indices_Ω = 1:length(Ω)


### Declaration du modele
model = Model(Gurobi.Optimizer)
#set_optimizer_attribute(model, "BarHomogeneous", 1)
#set_optimizer_attribute(model, "Method", 3)
#set_optimizer_attribute(model, "NumericFocus", 1)
#set_optimizer_attribute(model, "ScaleFlag", 2)


### Declaration des variables
@variable(model, x[v = indices_Vˢ, s = indices_S], binary=true, start=0)
@variable(model, y⁰[e ∈ indices_E⁰, q ∈ indices_Q⁰], binary=true, start=0)
@variable(model, yˢ[e = indices_Eˢ, q = indices_Qˢ], binary=true, start=0)
@variable(model, z[v = indices_Vˢ, t = indices_Vᵗ], binary=true, start=0)


file_path = joinpath(@__DIR__, "sol_small_heur.json")
sol_heur = JSON.parsefile(file_path)

for sub ∈ sol_heur["substations"]
    set_start_value(x[sub["id"], sub["substation_type"]], 1)
    set_start_value(y⁰[sub["id"], sub["land_cable_type"]], 1)
end

for cable ∈ sol_heur["substation_substation_cables"]
    for ind_e ∈ indices_Eˢ
        if cable["substation_id"] ∈ Eˢ[ind_e] && cable["other_substation_id"] ∈ Eˢ[ind_e]
            set_start_value(yˢ[ind_e, cable["cable_type"]], 1)
            break
        end
    end
end

for turb ∈ sol_heur["turbines"]
    set_start_value(z[turb["substation_id"], turb["id"]], 1)
end



# Solution([1, 1, 1], [0 0; 0 0], SubStation[SubStation(1, 1, 1)])
#set_start_value.(z[2,:], 1)
#set_start_value(x[2, 1], 1)
#set_start_value(y⁰[2, 1], 1)


### Declaration des contraintes

# Contrainte (1)
for v ∈ indices_Vˢ
    @constraint(model, sum(x[v, s] for s ∈ indices_S) <= 1)
end

# Contrainte (2)
for v ∈ indices_Vˢ
    @constraint(model, sum(y⁰[v, q] for q ∈ indices_Q⁰) - sum(x[v, s] for s ∈ indices_S) == 0)
end

# Contrainte (3)
for t ∈ indices_Vᵗ
    @constraint(model, sum(z[v, t] for v ∈ indices_Vˢ) == 1)
end

# Contrainte (4)
for v ∈ indices_Vˢ
    @constraint(model, sum(yˢ[e, q] for e ∈ indices_Eˢ, q ∈ indices_Qˢ if v ∈ Eˢ[e]) - sum(x[v, s] for s ∈ indices_S) <= 0)
end


### Creation de la fonction objectif

function linearize_product(α, β, m, M)
    # On suppose ici que β ∈ [m, M]
    μ = @variable(model)

    @constraint(model, μ - M*α <= 0)
    @constraint(model, μ - β + M*(1 - α) >= 0)
    @constraint(model, μ + α*m >= 0)
    @constraint(model, μ - β + m*(1 - α)<= 0)

    return μ
end

function linearize_positive_part(β, M)
    α = @variable(model, binary=true)
    
    @constraint(model, α * M - β <= M)
    @constraint(model, α * M - β >= 0)

    return linearize_product(α, β, M, M)
end

function linearize_min(α, β, M)
    return -linearize_positive_part(β - α, 2*M) + β
end

function linearize_positive_part_min_pb(β)
    μ = @variable(model)

    @constraint(model, μ >= 0)
    @constraint(model, μ >= β)

    return μ
end

# Construction de Cⁿ
max_rating_S = maximum([S[s]["rating"] for s ∈ indices_S])
max_rating_Q⁰ = maximum([Q⁰[q]["rating"] for q ∈ indices_Q⁰])
max_rating_S_Q⁰ = max(max_rating_S, max_rating_Q⁰)
list_min = [linearize_min(sum(S[s]["rating"] * x[v, s] for s ∈ indices_S), sum(Q⁰[q]["rating"] * y⁰[v, q] for q ∈ indices_Q⁰), max_rating_S_Q⁰) for v ∈ indices_Vˢ]

Cⁿ = [sum(linearize_positive_part_min_pb(Ω[ω]["power_generation"] * sum(z[v, t] for t ∈ indices_Vᵗ) - list_min[v]) for v ∈ indices_Vˢ) for ω ∈ indices_Ω]

# Contraintes pour ameliorer la relaxation continue du branch and bound
borne_Cⁿ = [Ω[ω]["power_generation"]*length(Vᵗ) for ω ∈ indices_Ω]
#@constraint(model, Cⁿ .>= 0)
#@constraint(model, Cⁿ .<= borne_Cⁿ)

# Construction de Cᶠ
max_rating_Qˢ = maximum([Qˢ[q]["rating"] for q ∈ indices_Qˢ])

Cᶠ = [linearize_positive_part_min_pb(Ω[ω]["power_generation"] * sum(z[v, t] for t ∈ indices_Vᵗ) - sum(Qˢ[q]["rating"] * yˢ[ẽ, q] for ẽ ∈ indices_Eˢ, q ∈ indices_Qˢ if v ∈ Eˢ[ẽ])) +  
      sum(linearize_positive_part_min_pb(Ω[ω]["power_generation"] * sum(z[(Eˢ[ẽ][1]!=v ? Eˢ[ẽ][1] : Eˢ[ẽ][2]), t] for t ∈ indices_Vᵗ) + linearize_min(sum(Qˢ[q]["rating"]*yˢ[ẽ, q] for q ∈ indices_Qˢ), Ω[ω]["power_generation"] * sum(z[v, t] for t ∈ indices_Vᵗ), max(max_rating_Qˢ, Ω[ω]["power_generation"] * length(Vᵗ))) - list_min[(Eˢ[ẽ][1]!=v ? Eˢ[ẽ][1] : Eˢ[ẽ][2])]) for ẽ ∈ indices_Eˢ if v ∈ Eˢ[ẽ])
      for v ∈ indices_Vˢ, ω ∈ indices_Ω]

borne_Cᶠ = [(Ω[ω]["power_generation"]*length(Vᵗ) + min(max_rating_Qˢ, Ω[ω]["power_generation"]*length(Vᵗ))) * length(Vˢ) for v ∈ indices_Vˢ, ω ∈ indices_Ω]
#@constraint(model, Cᶠ .>= 0)
#@constraint(model, Cᶠ .<= borne_Cᶠ)

# Construction de Cᶜ(Cⁿ) et Cᶜ(Cᶠ)
c₀ = data["general_parameters"]["curtailing_cost"]
cₚ = data["general_parameters"]["curtailing_penalty"]
Cₘₐₓ = data["general_parameters"]["maximum_curtailing"]

Cᶜ_Cⁿ = [c₀*Cⁿ[ω] + cₚ*linearize_positive_part_min_pb(Cⁿ[ω]-Cₘₐₓ) for ω ∈ indices_Ω]
Cᶜ_Cᶠ = [c₀*Cᶠ[v, ω] + cₚ*linearize_positive_part_min_pb(Cᶠ[v, ω]-Cₘₐₓ) for v ∈ indices_Vˢ, ω ∈ indices_Ω]

borne_Cᶜ_Cⁿ = [c₀ * borne_Cⁿ[ω] + cₚ * max(0, borne_Cⁿ[ω] - Cₘₐₓ) for ω ∈ indices_Ω]
borne_Cᶜ_Cᶠ = [c₀ * borne_Cᶠ[v, ω] + cₚ * max(0, borne_Cᶠ[v, ω] - Cₘₐₓ) for v ∈ indices_Vˢ, ω ∈ indices_Ω]
#@constraint(model, Cᶜ_Cⁿ - c₀*Cⁿ .>= 0)
#@constraint(model, Cᶜ_Cᶠ - c₀*Cᶠ.>= 0)
#@constraint(model, Cᶜ_Cⁿ .<= borne_Cᶜ_Cⁿ)
#@constraint(model, Cᶜ_Cᶠ .<= borne_Cᶜ_Cᶠ)

# Construction de la fonction objectif C
coords_v₀ = [data["general_parameters"]["main_land_station"]["x"], data["general_parameters"]["main_land_station"]["y"]]

c = sum(S[s]["cost"]*x[v, s] for v ∈ indices_Vˢ, s ∈ indices_S) +
    sum((Q⁰[q]["fixed_cost"] + norm([Vˢ[v]["x"], Vˢ[v]["y"]] - coords_v₀) * Q⁰[q]["variable_cost"]) * y⁰[v, q] for v ∈ indices_Vˢ, q ∈ indices_Q⁰) + 
    sum((Qˢ[q]["fixed_cost"] + norm([Vˢ[Eˢ[e][1]]["x"], Vˢ[Eˢ[e][1]]["y"]] - [Vˢ[Eˢ[e][2]]["x"], Vˢ[Eˢ[e][2]]["y"]]) * Qˢ[q]["variable_cost"]) * yˢ[e, q] for e ∈ indices_Eˢ, q ∈ indices_Qˢ) + 
    sum((data["general_parameters"]["fixed_cost_cable"] + norm([Vˢ[v]["x"], Vˢ[v]["y"]] - [Vᵗ[t]["x"], Vᵗ[t]["y"]]) * data["general_parameters"]["variable_cost_cable"]) * z[v, t] for v ∈ indices_Vˢ, t ∈ indices_Vᵗ) + 
    sum(Ω[ω]["probability"] * (sum(sum(S[s]["probability_of_failure"] * 1000 * (linearize_product(x[v, s], Cᶜ_Cᶠ[v, ω]/1000, 0, borne_Cᶜ_Cᶠ[v, ω]/1000) - linearize_product(x[v, s], Cᶜ_Cⁿ[ω]/1000, 0, borne_Cᶜ_Cⁿ[ω]/1000)) for s ∈ indices_S) + sum(Q⁰[q]["probability_of_failure"] * 1000 * (linearize_product(y⁰[v, q], Cᶜ_Cᶠ[v, ω]/1000, 0, borne_Cᶜ_Cᶠ[v, ω]/1000) - linearize_product(y⁰[v, q], Cᶜ_Cⁿ[ω]/1000, 0, borne_Cᶜ_Cⁿ[ω]/1000)) for q ∈ indices_Q⁰) for v ∈ indices_Vˢ) + Cᶜ_Cⁿ[ω]) for ω ∈ indices_Ω)
#@constraint(model, c >= 0)

# Declaration de la fonction objectif
@objective(model, Min, c)


### Resolution du probleme de minimisation
optimize!(model)


### Stockage de la solution
#using JLD2
#@save "small-sol.jld2" x=value.(x) y⁰=value.(y⁰) yˢ=value.(yˢ) z=value.(z)