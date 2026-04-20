using DataFrames, CSV
# ARGS = ["australia.soybean.txt", "0.1"]
# ARGS = ["henderson.milkfat.txt", "0.1"]
# ARGS = ["yates.oats.txt", "0.1"]
# ARGS = ["archbold.apple.txt", "0.1"]
# ARGS = ["acorsi.grayleafspot.txt", "0.1"]

"""
    prep_agridat_data(ARGS::Vector{String})::Nothing

Prepare agricultural dataset by identifying explanatory and response variables, 
and generating separate TSV files for each response variable.

# Arguments
- `ARGS::Vector{String}`: Command-line arguments where:
  - `ARGS[1]`: Path to input CSV file
  - `ARGS[2]`: Threshold ratio for determining if numeric columns are explanatory 
    (value between 0 and 1; columns are explanatory if unique values < threshold × nrow)

# Description
This function processes an agricultural dataset by:
1. Reading a CSV file, treating empty strings and various NA representations as missing
2. Classifying columns as explanatory or response variables based on:
   - Column names matching predefined lists
   - Data type (numeric vs. categorical)
   - Unique value count threshold for numerics
3. Converting numeric explanatory variables to categorical (prefixed with column name)
4. Filtering out rows with missing/NaN/Inf values in response variables
5. Creating separate TSV output files for each response variable with all explanatory variables

# Output
- Writes TSV files with pattern `{original_filename}-{response_variable_name}.tsv`
- Prints path of each generated output file to stdout
- Returns `nothing`

# Recognized Variable Names
- **Explanatory**: gen, pop, var, entry, env, year, loc, harvest, season, plot, rep, 
  row, col, blk, genotype, population, variety, cultivar, replication, column, block, 
  pos, position, spacing, stock, trt, treatment (and plural forms)
- **Response**: yield, grain, straw, height, size, lodging, protein, oil
"""
function prep_agridat_data(ARGS::Vector{String})::Nothing
    fname = ARGS[1]
    threshold_for_explanatory_numerics = parse(Float64, ARGS[2])
    df = CSV.read(fname, DataFrame, missingstring=["", "NA", "NAN", "NaN", "na", "nan"])
    potential_explanatory_names::Vector{String} = [
        "gen", "gens",
        "pop", "pops",
        "var", "vars",
        "entry", "entries",
        "env", "envs",
        "year", "years",
        "loc", "locs",
        "harvest", "harvests",
        "season", "seasons",
        "plot", "plots",
        "rep", "reps",
        "row", "rows",
        "col", "cols",
        "blk", "blks",
        "genotype", "genotypes",
        "population", "populations",
        "variety", "varieties",
        "cultivar", "cultivars",
        "replication", "replications",
        "column", "columns",
        "block", "blocks",
        "pos", "position", "positions",
        "spacing", "spacings",
        "stock", "stocks",
        "trt", "trts",
        "treatment", "treatments",
    ]
    potential_response_names::Vector{String} = [
        "yield",
        "grain",
        "straw",
        "height",
        "size",
        "lodging",
        "protein",
        "oil",
    ]
    # Identity explanatory and response variables
    (explanatory_names, response_names) = let
        explanatory_names::Vector{String} = []
        response_names::Vector{String} = []
        for j in 1:ncol(df)
            # j = 4
            id = names(df)[j]
            col = df[:, j]
            if id ∈ potential_explanatory_names
                # If the explanatory variable but it is not supposed to, i.e. all elements of potential_explanatory_names are assumed to be categorical
                # then we convert the numerics into categoricals
                if isa(col, Vector) && isa(col[1], Number)
                    df[!, id] = string.(id, "|", df[!, id])
                end
                push!(explanatory_names, id)
            elseif id ∈ potential_response_names
                if isa(col, Vector) && isa(col[1], Number)
                    push!(response_names, id)
                else
                    # We expect the response variables to be numeric if they are not then we skip
                    continue
                end
            elseif isa(col, Vector) # Numerics are Vectors in DataFrames
                if length(unique(col)) < threshold_for_explanatory_numerics*nrow(df) # likely not a response variable because of the limited (controlled by threshold_for_explanatory_numerics) number of unique values
                    push!(explanatory_names, id)
                else
                    push!(response_names, id)
                end
            else # Strings are not Vectors in DataFrames
                push!(explanatory_names, id)
            end
        end
        # In cases where there are no response variables detected because the response variable is numeric but with limited number of unique values,
        # then we arbitrarily relax the `threshold_for_explanatory_numerics` so that `threshold_for_explanatory_numerics*nrow(df) == 10`
        if length(response_names) == 0
            explanatory_names_repeat::Vector{String} = []
            response_names_repeat::Vector{String} = []
            RELAXED_THRESHOLD = 10
            for j in 1:ncol(df)
                # j = 4
                id = names(df)[j]
                col = df[:, j]
                if id ∈ potential_explanatory_names
                    # If the explanatory variable but it is not supposed to, i.e. all elements of potential_explanatory_names are assumed to be categorical
                    # then we convert the numerics into categoricals
                    if isa(col, Vector) && isa(col[1], Number)
                        df[!, id] = string.(id, "|", df[!, id])
                    end
                    push!(explanatory_names_repeat, id)
                elseif id ∈ potential_response_names
                    if isa(col, Vector) && isa(col[1], Number)
                        push!(response_names_repeat, id)
                    else
                        # We expect the response variables to be numeric if they are not then we skip
                        continue
                    end
                elseif isa(col, Vector) # Numerics are Vectors in DataFrames
                    if length(unique(col)) < RELAXED_THRESHOLD
                        push!(explanatory_names_repeat, id)
                    else
                        push!(response_names_repeat, id)
                    end
                else # Strings are not Vectors in DataFrames
                    push!(explanatory_names_repeat, id)
                end
            end
            (explanatory_names_repeat, response_names_repeat)
        else
            (explanatory_names, response_names)
        end
    end

    # Subset the data so that the first column corresponds to a single response variable and the rest are the explanatory variables
    if (length(explanatory_names) == 0) || (length(response_names) == 0)
        # No explanatory or response variables detected automatically just emit an empty string
        println("")
        return nothing
    else
        # Save each dataset with one response variable each
        for y_name in response_names
            # y_name = response_names[1]
            idx::Vector{Int64} = findall(.!ismissing.(df[!, y_name]) .&& .!isnan.(df[!, y_name]) .&& .!isinf.(df[!, y_name]))
            df_sub::DataFrame = select(df, vcat([y_name], explanatory_names))[idx, :]
            # We use the length of the explanatory_names as marker for where to start the indices of the response variable for mlp
            fname_out_tsv = string(join(split(fname, ".")[1:(end-1)], "."), "-", y_name, ".tsv")
            CSV.write(fname_out_tsv, df_sub, delim="\t")
            # println("explanatory_names: $explanatory_names")
            # println("response_names: $response_names")
            println(fname_out_tsv)
        end
    end
    nothing
end
# ARGS = ["archbold.apple.txt", "0.1"]
# @time @code_warntype prep_agridat_data(ARGS) # 2.515861 seconds (4.81 M allocations: 236.168 MiB, 3.24% gc time, 98.16% compilation time: 28% of which was recompilation)
# @time prep_agridat_data(ARGS) # 0.002661 seconds (7.55 k allocations: 4.291 MiB)

# ARGS = ["archbold.apple.txt", "0.1"]
# @time precompile(prep_agridat_data, (Vector{String}, )) # 2.798672 seconds (6.79 M allocations: 349.501 MiB, 14.29% gc time, 100.00% compilation time)
# @time prep_agridat_data(ARGS) # 2.483412 seconds (4.81 M allocations: 236.225 MiB, 2.00% gc time, 99.75% compilation time: 26% of which was recompilation)
# @time prep_agridat_data(ARGS) # 0.002301 seconds (7.55 k allocations: 4.291 MiB)

prep_agridat_data(ARGS)