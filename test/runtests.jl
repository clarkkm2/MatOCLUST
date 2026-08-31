using Test, MatOCLUST, Distributions, LinearAlgebra, StatsBase

r=200
n=2
p=4
G=2

M1=[−2.60 −1.10 −0.50 −0.20; 1.30 0.60 0.30 0.10]
M2=[1.50 1.70 1.90 2.20; -3.70 -2.70 -2.00 -1.50]
M=cat(M1, M2,dims=3)

U1=Symmetric([2.00 0.00; 0.00 1.00])
U2=Symmetric([1.70 0.50; 0.50 1.30])
U=cat(U1, U2, dims=3)


V1=Symmetric([1.00 0.50 0.25 0.13; 0.50 1.00 0.50 0.25; 0.25 0.50 1.00 0.50; 0.13 0.25 0.50 1.00])
V2=Symmetric([1.00 0.50 0.25 0.13; 0.50 1.00 0.50 0.25; 0.25 0.50 1.00 0.50; 0.13 0.25 0.50 1.00])
V=cat(V1, V2, dims=3)

class=sample([1,2],Weights([0.5,0.5]),r)

X=[rand(MatrixNormal(M[:,:,g],U[:,:,g],V[:,:,g]),1)[1] for g in class]

replaced=sample(1:r,10,replace=false)

for j in 1:10
    noise=rand(Uniform(-15,15),2)
    selected=sample(1:4,1)
    X[replaced[j]][:,selected]=noise
end

result = matOCLUST(X, 2, 20, maxiter = 1000, parallel = true, nsuccess = 100)

@test haskey(result, :bestmod)
@test haskey(result, :classes)
@test haskey(result, :numO)
@test haskey(result, :outs)
@test haskey(result, :KL)
@test length(result[:classes]) == length(X)
@test result[:numO] >= 0
@test length(result[:KL]) > 0