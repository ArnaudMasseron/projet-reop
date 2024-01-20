using JSON, Shuffle, Setfield

include("src/utils.jl")
include("src/instance.jl")
include("src/solution.jl")
include("src/parsing.jl")
include("src/eval.jl")

export Instance, Solution
export read_instance, read_solution, write_solution
export construction_cost, operational_cost, cost
export is_feasible


## Classes et fonctions

function mysetdif(A, max_Ω)
    """Calcule le complemntaire de A ⊂ Ω dans Ω = 1:max_Ω"""
    B = sort(A)
    Aᶜ=Vector{Int}()

    current_val = 1
    for val_b ∈ B
        while current_val < val_b
            push!(Aᶜ, current_val)
            current_val += 1
        end
        current_val += 1
    end
    Aᶜ = vcat(Aᶜ, B[end]+1:max_Ω)

    return Aᶜ
end


## Parametres
instance = read_instance(joinpath(@__DIR__, "instances/KIRO-large.json"))

Tᵢ = - 5000 / log(0.5)
Tₘᵢₙ = 5
α = 0.9
iterations_level = 3000


## Main
cheapest_sub_type_id = argmin([sub_type.cost for sub_type ∈ instance.substation_types])

# On prend le cable avec le variable cost le plus eleve parmis ceux qui ont le fixed cost le plus eleve
nb_substation_loc = length(instance.substation_locations)

min_fixed_cost = minimum([cab_type.fixed_cost for cab_type ∈ instance.land_substation_cable_types])
ids_min_fixed_cost = [cab_type.id for cab_type ∈ instance.land_substation_cable_types if cab_type.fixed_cost == min_fixed_cost]
cheapest_land_sub_cable_type_id = ids_min_fixed_cost[argmin([cab_type.variable_cost for cab_type ∈ instance.land_substation_cable_types if cab_type.fixed_cost == min_fixed_cost])]

min_fixed_cost = minimum([cab_type.fixed_cost for cab_type ∈ instance.substation_substation_cable_types])
ids_min_fixed_cost = [cab_type.id for cab_type ∈ instance.substation_substation_cable_types if cab_type.fixed_cost == min_fixed_cost]
cheapest_sub_sub_cable_type_id = ids_min_fixed_cost[argmin([cab_type.variable_cost for cab_type ∈ instance.substation_substation_cable_types if cab_type.fixed_cost == min_fixed_cost])]

nearest_sub_id = argmin(distance(sub, instance.land) for sub ∈ instance.substation_locations)

current_sol = Solution(;
    turbine_links = [nearest_sub_id for turb_loc ∈ instance.wind_turbines],
    inter_station_cables = zeros(Int, nb_substation_loc, nb_substation_loc),
    substations = [SubStation(; id=nearest_sub_id, substation_type=cheapest_sub_type_id, land_cable_type=cheapest_land_sub_cable_type_id)]
)

iteration_counter = 1

while Tᵢ > Tₘᵢₙ
    global current_sol
    global iteration_counter


    ## Exploration du voisinage substation V₁
    sol_candidate = deepcopy(current_sol)
    p_choose_swap = length(current_sol.substations)*(nb_substation_loc-length(current_sol.substations)) / (length(current_sol.substations)*(nb_substation_loc-length(current_sol.substations)) + nb_substation_loc)
    do_we_swap = (rand()<p_choose_swap ? true : false)

    if do_we_swap
        built_sub_ids = [sub.id for sub ∈ current_sol.substations]
        empty_sub_ids = [id for id ∈ mysetdif(getindex.(built_sub_ids, 1), nb_substation_loc)] #mysetdif a une complexite O(NlogN) avec N = length(current_sol.substations)

        old_sub = rand(current_sol.substations)
        new_loc_id = rand(empty_sub_ids)
        sub_type = old_sub.substation_type
        cab_type = old_sub.land_cable_type

        # On detruit une station, on la reconstruit autre part
        deleteat!(sol_candidate.substations, findall(sub->sub.id==old_sub.id, sol_candidate.substations))
        push!(sol_candidate.substations, SubStation(; id=new_loc_id, substation_type=sub_type, land_cable_type=cab_type))

        # On trensfere une eventuelle liaison substation-substation au nouvel emplacement
        eventual_link = findall(x->x!=0, sol_candidate.inter_station_cables[old_sub.id,:]) # complexite O(N) avec N = length(current_sol.substations)
        if !isempty(eventual_link)
            global sol_candidate
            j = eventual_link[1]
            cab_type = sol_candidate.inter_station_cables[old_sub.id, j]
            sol_candidate.inter_station_cables[old_sub.id, j] = sol_candidate.inter_station_cables[j, old_sub.id] = 0
            sol_candidate.inter_station_cables[new_loc_id, j] = sol_candidate.inter_station_cables[j, new_loc_id] = cab_type
        end

        # On transfere des eventuelles liaisons a des wind turbines
        eventual_link = findall(x->x==old_sub.id, current_sol.turbine_links) # complexite O(N) avec N = length(current_sol.turbine_links)
        for turb_id ∈ eventual_link
            global sol_candidate
            sol_candidate.turbine_links[turb_id] = new_loc_id
        end
    else
        # On detruit ou reconstruit une substation
        sub_id = rand(1:nb_substation_loc)

        if isempty(findall(sub->sub.id==sub_id, sol_candidate.substations))
            # Construction d'une nouvelle substation
            push!(sol_candidate.substations, SubStation(; id=sub_id, substation_type=cheapest_sub_type_id, land_cable_type=cheapest_land_sub_cable_type_id))
        elseif length(sol_candidate.substations) > 1 # On ne veut pas qu'il y ait 0 substations apres la manip
            # Destruction d'une substation
            global sol_candidate
            deleteat!(sol_candidate.substations, findall(sub->sub.id==sub_id, sol_candidate.substations))

            # On detruit un eventuel cable qui etait relie a une autre substation
            eventual_link = findall(x->x!=0, sol_candidate.inter_station_cables[sub_id,:])
            if !isempty(eventual_link)
                global sol_candidate
                j = findall(x->x!=0, sol_candidate.inter_station_cables[sub_id,:])[1]
                sol_candidate.inter_station_cables[sub_id, j] = sol_candidate.inter_station_cables[j, sub_id] = 0
            end

            # Les turbines qui etaient reliees a cette substation vont etre reliees a la substation construite la plus proche
            eventual_links = findall(x->x==sub_id, current_sol.turbine_links)
            for turb_id ∈ eventual_links
                global sol_candidate
                sol_candidate.turbine_links[turb_id] = sol_candidate.substations[argmin([distance(instance.wind_turbines[turb_id], instance.substation_locations[sub.id])] for sub ∈ sol_candidate.substations)].id
            end
        end
    end

    #@assert is_feasible(sol_candidate, instance)
    current_sol_cost = cost(current_sol, instance)
    ∆f = cost(sol_candidate, instance) - current_sol_cost
    if ∆f < 0  || rand() < exp(-∆f/Tᵢ)
        global current_sol
        current_sol = sol_candidate
    end

    ## Exploration du voisinage turbine links V₂
    sol_candidate = deepcopy(current_sol)

    turb_id = rand(1:nb_turbines(instance)) # complexite O(|Vᵗ|)
    sub_id = rand(current_sol.substations).id # complexite O(|Vˢ|)

    sol_candidate.turbine_links[turb_id] = sub_id

    #@assert is_feasible(sol_candidate, instance)
    current_sol_cost = cost(current_sol, instance)
    ∆f = cost(sol_candidate, instance) - current_sol_cost
    if ∆f < 0  || rand() < exp(-∆f/Tᵢ)
        global current_sol
        current_sol = sol_candidate
    end


    ## Exploration du voisinage cable inter substation V₃
    sol_candidate = deepcopy(current_sol)

    built_cables = Vector{Tuple{Int, Int, Bool}}()
    non_connected_sub_ids = Vector{Int}()
    # Complexite de cette boucle for: O(|Vˢ|²)
    for sub ∈ current_sol.substations
        i = sub.id
        sub_linked = findall(x->x!=0, current_sol.inter_station_cables[i,:])

        if isempty(sub_linked)
            push!(non_connected_sub_ids, i)
        elseif sub_linked[1] > i
            push!(built_cables, (i, sub_linked[1], true))
        end
    end

    possible_constructions = Vector{Tuple{Int, Int, Bool}}()
    # Complexite de cette boucle for: O(|Vˢ|²)
    for i ∈ 1:length(non_connected_sub_ids)
        for j ∈ i+1:length(non_connected_sub_ids)
            push!(possible_constructions, (non_connected_sub_ids[i], non_connected_sub_ids[j], false))
        end
    end

    # On peut enlever un cable et le mettre a un endroit vide
    possible_swaps = vcat([(i, j) for i ∈ 1:length(built_cables), j ∈ 1:length(possible_constructions)]...) # Complexite O(|Vˢ|²)

    # On peut aussi déconnecter un able d'une substation et le connecter a une autre substation
    possible_swaps = vcat(possible_swaps, vcat([((cab[1], cab[2]), sub) for cab ∈ built_cables, sub ∈ non_connected_sub_ids]...), vcat([((cab[2], cab[1]), sub) for cab ∈ built_cables, sub ∈ non_connected_sub_ids]...)) # Complexite O(|Vˢ|²)

    V₂ = vcat(built_cables, possible_constructions, possible_swaps)

    if !isempty(V₂)
        x = rand(V₂)

        if isa(x, Tuple{Int, Int, Bool})
            if x[3] == true
                # On detruit un cable
                global sol_candidate
                sol_candidate.inter_station_cables[x[1], x[2]] = sol_candidate.inter_station_cables[x[2], x[1]] = 0
            else
                # On construit un cable
                global sol_candidate
                sol_candidate.inter_station_cables[x[1], x[2]] = sol_candidate.inter_station_cables[x[2], x[1]] = cheapest_sub_sub_cable_type_id
            end
        elseif isa(x, Tuple{Tuple{Int, Int}, Int})
            # On deconnecte un cable d'une des substation et on le relie a une autre
            cab_type = sol_candidate.inter_station_cables[x[1][1], x[1][2]]
            sol_candidate.inter_station_cables[x[1][1], x[1][2]] = sol_candidate.inter_station_cables[x[1][2], x[1][1]] = 0
            sol_candidate.inter_station_cables[x[1][1], x[2]] = sol_candidate.inter_station_cables[x[2], x[1][1]] = cab_type
        else
            # On enleve un cable pour le mettre entre deux autres substations differentes des substations initiales
            global sol_candidate
            og_loc = built_cables[x[1]]
            new_loc = possible_constructions[x[2]]
            cab_type = sol_candidate.inter_station_cables[og_loc[1], og_loc[2]]
            sol_candidate.inter_station_cables[og_loc[1], og_loc[2]] = sol_candidate.inter_station_cables[og_loc[2], og_loc[1]] = 0
            sol_candidate.inter_station_cables[new_loc[1], new_loc[2]] = sol_candidate.inter_station_cables[new_loc[2], new_loc[1]] = cab_type
        end
    end

    #@assert is_feasible(sol_candidate, instance)
    current_sol_cost = cost(current_sol, instance)
    ∆f = cost(sol_candidate, instance) - current_sol_cost
    if ∆f < 0  || rand() < exp(-∆f/Tᵢ)
        global current_sol
        current_sol = sol_candidate
    end
    

    ## Exploration voisinage substation type V₄
    sol_candidate = deepcopy(current_sol)

    x = (rand(1:length(current_sol.substations)), rand(instance.substation_types).id) # Complexite O(|Vˢ|+|S|)
    
    current_substation = sol_candidate.substations[x[1]]
    sol_candidate.substations[x[1]] = @set current_substation.substation_type = x[2]

    #@assert is_feasible(sol_candidate, instance)
    current_sol_cost = cost(current_sol, instance)
    ∆f = cost(sol_candidate, instance) - current_sol_cost
    if ∆f < 0  || rand() < exp(-∆f/Tᵢ)
        global current_sol
        current_sol = sol_candidate
    end


    ## Exploration voisinage substation substation cable type V₅
    sol_candidate = deepcopy(current_sol)

    built_cables = Vector{Tuple{Int, Int}}()
    # Complexite O(|Vˢ|²)
    for i ∈ 1:nb_substation_loc
        sub_linked = findall(x->x!=0, current_sol.inter_station_cables[i,:])

        if !isempty(sub_linked) && sub_linked[1] > i
            push!(built_cables, (i, sub_linked[1]))
        end
    end

    if !isempty(built_cables)
        global sol_candidate
        x = (rand(built_cables), rand(instance.substation_substation_cable_types).id) # Complexite O(|V^s| + |Qˢ|)
        sol_candidate.inter_station_cables[x[1][1], x[1][2]] = sol_candidate.inter_station_cables[x[1][2], x[1][1]] = x[2]
        
        #@assert is_feasible(sol_candidate, instance)
        current_sol_cost = cost(current_sol, instance)
        ∆f = cost(sol_candidate, instance) - current_sol_cost
        if ∆f < 0  || rand() < exp(-∆f/Tᵢ)
            global current_sol
            current_sol = sol_candidate
        end
    end


    ## Exploration voisinage land substation cable type V₆
    sol_candidate = deepcopy(current_sol)

    x = (rand(1:length(current_sol.substations)), rand(instance.land_substation_cable_types).id) # Complexite en O(|Vˢ|+|Q⁰|)
    current_substation = sol_candidate.substations[x[1]]
    sol_candidate.substations[x[1]] = @set current_substation.substation_type = x[2]

    #@assert is_feasible(sol_candidate, instance)
    current_sol_cost = cost(current_sol, instance)
    ∆f = cost(sol_candidate, instance) - current_sol_cost
    if ∆f < 0  || rand() < exp(-∆f/Tᵢ)
        global current_sol
        current_sol = sol_candidate
    end

    ## Baisse de la temperature
    if iteration_counter == iterations_level
        global Tᵢ
        global iteration_counter
        global current_sol
        Tᵢ *= α
        iteration_counter = 0
        println(Tᵢ)
    end
    iteration_counter += 1
end

current_sol_cost = cost(current_sol, instance)
println(current_sol_cost)

#write_solution(current_sol, "KIRO-large-sol_20.json")