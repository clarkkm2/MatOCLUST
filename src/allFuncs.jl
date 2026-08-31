minus(indx, x) = setdiff(1:length(x), indx)

function makeSymmetric(A::Matrix{Float64})
    B=triu(A)+triu(A)'-Diagonal(diag(A))
    B=Symmetric(B)
    return(B)
end

function randSymmetric(p::Int64)
    eigvals=rand(Uniform(1,10),p)
    Q, _ = qr(randn(p, p))
    D = Diagonal(eigvals)
    A = Q*D*Q'
    return(A)
end

"""
Run matrix-variate normal clustering. Initialization options are kmeans, random soft, or user-specified matrix. Note: kmeans is modified to run for only 5 iterations.

# Arguments
- `X`: vector of matrices
- `G`: number of clusters
- `maxiter`: maximum number of EM iterations
- `eps`: convergence criterion
- `zinit`: initialization option
- `loglikonly`: a Boolean specifying if only the log-likelihood is desired

# Returns
A dictionary containing:
- `M`: vector of mean matrices
- `U`: vector of row covariance matrices
- `V`: vector of column covariance matrices
- `z`: matrix of posterior probabilities
- `loglik`: the final log-likelihood of the model
"""
function EM(X::Vector{Matrix{Float64}},G::Int64; maxiter::Int64=100,eps::Float64=1E-10,zinit="kmeans",loglikonly::Bool=false)

    N::Int64 = size(X,1);
    r::Int64 = size(X[1],1);
    c::Int64 = size(X[1],2);

    if (zinit isa Matrix)
        z=zinit
    else
        if (zinit=="kmeans")  
            kclass=kmeans(X,G,maxiter=5)
            z=zeros(N,G)
            inds=[map(x->x[1],findall(x->x==g,kclass)) for g in 1:G]
            for g in 1:G
                z[inds[g],g].=1
            end

        else
            z=rand(N,G)
            z=z./sum(z,dims=2)
        end
    end

    # Initilize V
    Vinit=[randSymmetric(c) for i in 1:G]
    
    # Calculate M and U
    M= [wsum(X,z[:,j])/sum(z[:,j]) for j in 1:G]
    detU, U =U_est(X,M,Vinit,z)

    trU=tr.(U)
    U=r*U./trU
    
    detU=(detU).*(r./trU).^r

    detV, V = V_est(X,M,U, z)
    
    # Perform first E-step
    ztry, loglikupdate = Esteps(X,M,U,detU,V,detV,z,G)

    loglik = [loglikupdate]

    if (any(isnan.(ztry)) || any(sum(ztry,dims=1).<2))
        z=rand(N,G)
        z=z./sum(z,dims=2)
    else
        z=ztry
    end
    
    tol=1E10
    iter=2

    while (tol>eps && iter<=maxiter)
        M= [wsum(X,z[:,j])/sum(z[:,j]) for j in 1:G]
        
        detU, U =U_est(X,M,V,z)
        trU=tr.(U)
        U=r*U./trU
        detU=(detU).*(r./trU).^r

        detV, V = V_est(X,M,U,z)
               
        ztry, loglikupdate =Esteps(X,M,U,detU,V,detV,z,G)

        if (any(isnan.(ztry)) || any(sum(ztry,dims=1).<2))
            z=rand(N,G)
            z=z./sum(z,dims=2)
            V=[randSymmetric(c) for i in 1:G]
            detV=det.(V)
        else
            z=ztry
            loglik=append!(loglik,loglikupdate)
            
            if (iter>1)
                tol=abs((loglik[iter]-loglik[(iter-1)]))
            end
            iter+=1
        end
        
    end

    up=Int64(round(iter*2/3))

    if (loglikonly==false)
        returnVar::Dict{Symbol,Any} = Dict{Symbol,Any}();
        returnVar[:M]=M
        returnVar[:U]=U
        returnVar[:V]=V
        returnVar[:z]=z
        returnVar[:loglik]=last(loglik)
        return(returnVar)
    else
        return(last(loglik))
    end
    
end

function Esteps(X::Vector{Matrix{Float64}}, M::Vector{Matrix{Float64}},U::Vector{Matrix{Float64}},detU::Vector{Float64},V::Vector{Matrix{Float64}},detV::Vector{Float64},z::Matrix{Float64},G::Int64)
    N=size(X,1)
    r=size(X[1],1)
    c=size(X[1],2)

    invU=inv.(U)
    invV=inv.(V)

    dens=zeros(N,G)
    for g in 1:G
        dens[:,g]=[-r*c*log(2*pi)/2-c/2*log(detU[g])-r/2*log(detV[g])-1/2*tr(invV[g]*(X[i]-M[g])'*invU[g]*(X[i]-M[g])) for i in 1:N]
        end

    Ng=sum(z,dims=1)
    pig=Ng/N
    k=maximum(dens)
    if (k>700)
        over=k-709
        k=over
    else
        k=0
    end
    ztmp=exp.(dens.-k)
    ztmp=ztmp.*pig
    zsum=sum(ztmp,dims=2)

    z=ztmp./zsum
    loglik=sum(log.(zsum).+k)
    
    return(z,loglik)
end

function U_est(X::Vector{Matrix{Float64}}, M::Vector{Matrix{Float64}},V::Vector{Matrix{Float64}}, z::Matrix{Float64})
    Ng=sum(z,dims=1);
    c::Int64 = size(X[1],2);
    r::Int64 = size(X[1],1);
    G::Int64=size(z,2)

    V_inv= inv.(V)
    U=[Matrix{Float64}(undef,r,r) for _ in 1:G]


    # Begin for loop
    for i in 1:G
        denom::Float64 = c*Ng[i];
        XML=[x-M[i] for x in X]
        Utmp=[y*V_inv[i]*y'/denom for y in XML]
        U[i]=wsum(z[:,i],Utmp)
    end

    detU=det.(U)

    if any(detU.<1e-18)
        for g in 1:G
            if (detU[g]<1e-18)
                U[g]=randSymmetric(r)
            end
        end
        detU=det.(U)
    end

    return(detU, U)
end

function V_est(X::Vector{Matrix{Float64}},M::Vector{Matrix{Float64}}, U::Vector{Matrix{Float64}}, z::Matrix{Float64})

    Ng=sum(z,dims=1);
    c::Int64 = size(X[1],2);
    r::Int64 = size(X[1],1);
    G::Int64=size(z,2)

    U_inv = inv.(U)
    V=[Matrix{Float64}(undef,c,c) for _ in 1:G]

    # Begin for loop
    for i in 1:G
        denom::Float64 = r*Ng[i];
        XML=[x-M[i] for x in X]
        Vtmp=[y'*U_inv[i]*y/denom for y in XML]
        V[i]=wsum(z[:,i],Vtmp)
    end
    
    detV=det.(V)
    
    if any(detV.<1e-18)
        for g in 1:G
            if (detV[g]<1e-18)
                V[g]=randSymmetric(c)
            end
        end
        detV=det.(V)
    end
    return(detV,V)
end

function mixgammacdf(x::Float64;pis::Matrix{Float64},r::Int64,c::Int64,Us::Vector{Matrix{Float64}},Vs::Vector{Matrix{Float64}})
    G=length(pis)
    cons=vec(-log.(pis).+r*c/2*log(2*pi)).+c/2*log.(det.(Us)).+r/2*log.(det.(Vs))
    cgamma=[pis[i]*map(z->cdf(Gamma((r*c/2),1), z-cons[i]),x) for i in 1:G]
    cgamma=sum(reduce(hcat,cgamma),dims=2)
    return(cgamma)
end 

function mixgammapdf(x::Float64;pis::Matrix{Float64},r::Int64,c::Int64,Us::Vector{Matrix{Float64}},Vs::Vector{Matrix{Float64}})
    G=length(pis)
    cons=vec(-log.(pis).+r*c/2*log(2*pi)).+c/2*log.(det.(Us)).+r/2*log.(det.(Vs))
    pgamma=[pis[i]*map(z->pdf(Gamma((r*c/2),1), z-cons[i]),x) for i in 1:G]
    pgamma=sum(reduce(hcat,pgamma),dims=2)
    return(pgamma)
end 

"""
Run multiple initializations of the EM algorithm. 
Runs the EM function for 10 iterations with a new initialization each time. 
After the specified number of successful runs are completed, the model continues the model with largest log-likelihood to convergence.

# Arguments
- `X`: vector of matrices
- `G`: number of clusters
- `maxiter`: maximum number of EM iterations
- `eps`: convergence criterion
- `zinit`: initialization option
- `nsuccess`: number of starts to try
- `loglikonly`: a Boolean specifying if only the log-likelihood is desired

# Returns
A dictionary containing:
- `M`: vector of mean matrices
- `U`: vector of row covariance matrices
- `V`: vector of column covariance matrices
- `z`: matrix of posterior probabilities
- `loglik`: the final log-likelihood of the model
"""
function miniEM(X::Vector{Matrix{Float64}},G::Int64;maxiter::Int64=100,eps::Float64=1E-10,zinit="kmeans", nsuccess::Int64 = 10, loglikonly::Bool=false)
    loglik=-Inf
    check=false
    ngood=0
    best = nothing
    it=1
    while (ngood<=nsuccess && it<10*nsuccess)
        test=EM(X,G,maxiter=10,eps=1E-5,zinit=zinit,loglikonly=false)
        check=(sum(isnan.(test[:z]))==0)
        if(check==true)
            ngood=ngood+1
        end
        if (last(test[:loglik])>loglik && check==true)
            loglik=last(test[:loglik])
            best=test
        end
        it=it+1
    end

    final=EM(X,G,maxiter=maxiter,eps=eps,zinit=best[:z],loglikonly=loglikonly)

    return(final)
end

function applyEM(i::Int64,N::Int64,X::Vector{Matrix{Float64}},G::Int64,maxiter::Int64,eps::Float64,zinit,loglikonly::Bool)
    zsub=zinit[minus(i,1:N),:]
    Xsub=X[minus(i,1:N)]
    loglik=EM(Xsub,G,maxiter=maxiter,eps=eps,zinit=zsub,loglikonly=loglikonly)
    return(loglik)
end

function  KL(shiftlik::Vector{Float64};r::Int64,c::Int64,pis::Matrix{Float64},Us::Vector{Matrix{Float64}},Vs::Vector{Matrix{Float64}})
    N=length(shiftlik)
    cons=vec(-log.(pis).+r*c/2*log(2*pi)).+c/2*log.(det.(Us)).+r/2*log.(det.(Vs))

    minlik=floor(minimum(append!(cons,filter(!isnan,shiftlik))))
    maxlik=ceil(maximum(filter(!isnan,shiftlik))) 
    edge=minlik:1:maxlik
    h = StatsBase.fit(Histogram, shiftlik, edge)
    edge=h.edges[1]
    nbin=length(h.weights)
    freqnull=[mixgammacdf(x,pis=pis,r=r,c=c,Us=Us,Vs=Vs)[1] for x in edge]
    freqnull=(freqnull[2:(nbin+1)]-freqnull[1:nbin]).+abs.(rand(Normal(0,1E-30),nbin))
    freq=(h.weights/N)
    KLDiv=kldivergence(freq,freqnull)
    return(KLDiv)
end

"""
A modified version of the kmeans algorithm that runs only for maxiter iterations-- not necessarily to convergence.

# Arguments
- `X`: vector of matrices
- `k`: number of clusters
- `maxiter`: maximum number of kmeans iterations

# Returns
A vector of classes
"""
function kmeans(X::Vector{Matrix{Float64}},k::Int64;maxiter::Int64=15)
    r = size(X,1)

    meanind = sample(1:r,k,replace=false)
    means = [X[meanind[g]] for g in 1:k]

    convcrit = false
    classold = zeros(Int,r)

    classes = zeros(Int,r)  

    niter = 1

    while !convcrit && niter <= maxiter
        diff = [[sum((X[i]-means[g]).^2) for i in 1:r] for g in 1:k]
        diff = transpose(mapreduce(permutedims, vcat, diff))

        classes = vec(mapslices(argmin, diff; dims=2))

        if classes == classold
            convcrit = true
        else
            classold = copy(classes)
        end

        inds = [findall(==(g), classes) for g in 1:k]
        means = [mean(X[inds[g]]) for g in 1:k]

        niter += 1
    end

    return classes
end

"""
Creates a confusion matrix.

# Arguments
- `predicted`: vector of predicted classes
- `truth`: vector of true classes

# Returns
A confusion matrix DataFrame
"""
function crosstab(predicted,truth)
    cat1=sort(unique(predicted))
    cat2=sort(unique(truth))

    crosstab = DataFrame([[] for i in 1:(length(cat1)+1)], pushfirst!(string.(cat1),"Pred Class"))

    for j in 1:length(cat2)
        push!(crosstab, pushfirst!([sum((predicted.==cat1[i]).&&(truth.==cat2[j])) for i in 1:length(cat1)],cat2[j]))
    end
    return(crosstab)
end

"""
Run matrix outlier clustering.

# Arguments
- `X`: vector of matrices
- `G`: number of clusters
- `F`: maximum number of outliers
- `maxiter`: maximum number of EM iterations
- `eps`: convergence criterion
- `zinit`: initialization option
- `nsuccess`: number of starts to try for each iteration of matOCLUST
- `parallel`: a Boolean specifying if subset log-likelihoods should be calculated in parallel
- `prnt`: a Boolean specifying if a progress bar should be displayed

# Returns
A dictionary containing:
- `bestmod`: the model corresponding to minimum KL
- `classes`: the classes from the best model
- `numO`: the number of outliers predicted
- `outs`: the numO outliers identified
- `allCand`: all possible outlier candidates-- possibly more than given by outs
- `KL`: a vector of KL divergence at each iteration, 0:F.
"""
function matOCLUST(X::Vector{Matrix{Float64}}, G::Int64, F::Int64; maxiter::Int64=1000, eps=1E-5,zinit="kmeans", nsuccess::Int64 = 10, parallel::Bool = true, prnt::Bool = true)
    absN=size(X)[1]
    r,c = size(X[1])
    newX=copy(X)
    outs=[]
    bestmod = Dict()
    classes = nothing
    numO = -1
    
    KLvec = []
    minKL = Inf

    indsLeft=1:absN
    
    results = Dict{Symbol, Any}()

    niter=0
    if prnt
        total = F+1
        pb = Progress(total;
                    desc = "",
                    barglyphs=BarGlyphs("[=> ]"))
    end

    if parallel && Threads.nthreads() == 1
        @warn "parallel=true but Julia was started with only one thread"
        parallel = false
    end

    while (niter<=F)
    
        relN=size(newX,1)

        #initialize EM a few times and run EM with best initialization
        mod = miniEM(newX,G,maxiter=maxiter,eps=eps,zinit=zinit, nsuccess = nsuccess, loglikonly=false)

        Us=mod[:U]
        Vs=mod[:V]
        Ms=mod[:M]
        z=mod[:z]

        pis=sum(z,dims=1)/relN
        
              
        subsetlogs = Vector{Float64}(undef, relN)

        if parallel
            Threads.@threads for j in 1:relN
                subsetlogs[j] = applyEM(j, relN, newX, G, 20, 1e-5, z, true)
            end
        else
            for j in 1:relN
                subsetlogs[j] = applyEM(j, relN, newX, G, 20, 1e-5, z, true)
            end
        end
       
        #Find shifted log-likelihoods and worst point
        shiftliks=subsetlogs.-last(mod[:loglik])
        relbad=argmax(shiftliks)
        absbad=indsLeft[relbad]
        outs=append!(outs,absbad)

        KLdist = KL(shiftliks,r=r,c=c,pis=pis,Us=Us,Vs=Vs)
        KLvec = append!(KLvec,KLdist)

        if KLdist < minKL
            minKL = KLdist
            bestmod = mod
            classes = zeros(Int, absN)
            classes[indsLeft] = getindex.(argmax(z, dims=2), 2)
            numO = niter
        end

        #Remove index of bad point
        indsLeft=setdiff(indsLeft,absbad)
        
        #create new dataset with points remaining
        newX=X[indsLeft] 
        if prnt
            next!(pb)
        end
        niter+=1        
    end 
    results[:bestmod]=bestmod
    results[:classes]=classes
    results[:numO] = numO
    if numO <1
        results[:outs] = []
    else
    results[:outs] = outs[1:(numO+1)]
    end
    results[:allCand] = outs
    results[:KL] = KLvec

    return(results)
end
 

