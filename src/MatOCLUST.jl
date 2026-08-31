module MatOCLUST

using Random
using Distributions
using LinearAlgebra
using DataFrames
using Statistics
using StatsBase
using ProgressMeter

export matOCLUST,
       EM,
       miniEM,
       kmeans,
       crosstab

include("allFuncs.jl")

end
