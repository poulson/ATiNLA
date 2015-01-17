# In[1]:

n = 1000;
useWilk = false;

if useWilk
    # Build Wilkinson's example (NOTE: the "Wilkinson" matrix is different)
    AOrig = tril(-ones(n,n),-1) + Diagonal(ones(n));
    AOrig[:,n] = -ones(n);
else
    AOrig = randn(n,n);
end

ANorm=norm(AOrig);
println("|| A ||_2 = $ANorm")
ACond=cond(AOrig);
println("cond(A) = $ACond")
# Build a random right-hand side
b = randn(n);
bNorm = norm(b);
println("|| b ||_2 = $bNorm")


# In[2]:

function LUUnb(A)
    m,n=size(A);
    for k=1:n
        A[k+1:end,k] /= A[k,k];
        A[k+1:end,k+1:end] -= A[k+1:end,k]*A[k,k+1:end];
    end
end

function LU(A,bsize)
    m, n = size(A);
    for k=1:bsize:n,
        nb=min(n-k+1,bsize);
        ind1 = k:k+nb-1;
        ind2 = k+nb:n;
        A11 = sub(A,ind1,ind1);
        A12 = sub(A,ind1,ind2);
        A21 = sub(A,ind2,ind1);
        A22 = sub(A,ind2,ind2);
        
        LUUnb(A11)

        # A21 := A21 inv(U11)
        BLAS.trsm!('R','U','N','N',1.,A11,A21);

        # A12 := inv(L11) A12
        BLAS.trsm!('L','L','N','U',1.,A11,A12);
        
        # A22 := A22 - A21*A12
        BLAS.gemm!('N','N',-1.,A21,A12,1.,A22)
    end
end

println("Starting unpivoted unblocked LU factorization")
A = copy(AOrig);
tic();
LUUnb(A);
unbTime=toq();
println("Unblocked time: $unbTime")
L = tril(A,-1) + Diagonal(ones(n));
U = triu(A);
decompError=norm(AOrig - L*U);
relDecompError=decompError/ANorm;
println("|| A - L U ||             = $decompError")
println("|| A - L U || / || A ||   = $relDecompError")
# A = L U implies inv(A) b = inv(L U) b = inv(U) inv(L) b
x = U\(L\b);
residError=norm(b-AOrig*x);
relResidError=residError/bNorm;
println("|| b - A x ||_2             = $residError")
println("|| b - A x ||_2 / || b ||_2 = $relResidError")

println("Starting unpivoted blocked LU factorization")
A = copy(AOrig);
tic();
LU(A,96);
blockTime=toq();
println("Blocked time: $blockTime")
L = tril(A,-1) + Diagonal(ones(n));
U = triu(A);
decompError=norm(AOrig - L*U);
relDecompError=decompError/ANorm;
println("|| A - L U ||             = $decompError")
println("|| A - L U || / || A ||   = $relDecompError")
# A = L U implies inv(A) b = inv(L U) b = inv(U) inv(L) b
x = U\(L\b);
residError=norm(b-AOrig*x);
relResidError=residError/bNorm;
println("|| b - A x ||_2             = $residError")
println("|| b - A x ||_2 / || b ||_2 = $relResidError")

# In[3]:

function LUPartialUnb(A)
    m,n=size(A);
    p = [1:n];
    for k=1:n
        # Search for the maximum entry in A(k:end,k)
        iPiv, pivVal = k, abs(A[k,k]);
        for i=k:n
            if abs(A[i,k]) > pivVal
                iPiv, pivVal = i, abs(A[i,k]);
            end
        end
    
        # Swap A[k,:] with A[iPiv,:] and accumulate the permutations in p
        A[[k,iPiv],:] = A[[iPiv,k],:];
        p[[k,iPiv]  ] = p[[iPiv,k]  ];
    
        A[k+1:end,k] /= A[k,k];
        A[k+1:end,k+1:end] -= A[k+1:end,k]*A[k,k+1:end];
    end
    return p
end

function LUPartialPanel(A)
    m,n=size(A);
    p=[1:m];
    for k=1:n
        # Search for the maximum entry in A(k:end,k)
        iPiv, pivVal = k, abs(A[k,k]);
        for i=k:m
            if abs(A[i,k]) > pivVal
                iPiv, pivVal = i, abs(A[i,k]);
            end
        end
    
        # Swap A[k,:] with A[iPiv,:] and accumulate the permutations in p
        A[[k,iPiv],:] = A[[iPiv,k],:];
        p[[k,iPiv]  ] = p[[iPiv,k]  ];
    
        A[k+1:end,k] /= A[k,k];
        A[k+1:end,k+1:end] -= A[k+1:end,k]*A[k,k+1:end];
    end
    return p;
end

function LUPartial(A,bsize)
    m, n = size(A);
    p = [1:n];
    for k=1:bsize:n,
        nb=min(n-k+1,bsize);
        ind0 = 1:k-1;
        ind1 = k:k+nb-1;
        ind2 = k+nb:n;
        indB = k:n;
        A11 = sub(A,ind1,ind1);
        A12 = sub(A,ind1,ind2);
        A21 = sub(A,ind2,ind1);
        A22 = sub(A,ind2,ind2);
        AB0 = sub(A,indB,ind0);
        AB1 = sub(A,indB,ind1);
        AB2 = sub(A,indB,ind2);
        pB = sub(p,indB);
        
        # Perform the pivoted panel factorization of AB = [A11;A21]
        pPan = LUPartialPanel(AB1);
        
        # Apply the permutations used for factoring AB = [A11;A21]
        AB0[:,:] = AB0[pPan,:];
        AB2[:,:] = AB2[pPan,:];
        pB[:] = pB[pPan];
                
        # A12 := inv(L11) A12
        BLAS.trsm!('L','L','N','U',1.,A11,A12);
        
        # A22 := A22 - A21*A12
        BLAS.gemm!('N','N',-1.,A21,A12,1.,A22)
    end
    return p
end

# Run the unblocked code
println("Starting unblocked partially-pivoted LU")
A = copy(AOrig);
tic();
p = LUPartialUnb(A);
unbTime = toq();
println("Unblocked time: $unbTime")
L = tril(A,-1) + Diagonal(ones(n));
U = triu(A);
decompError=norm(AOrig[p,:]-L*U);
relDecompError=decompError/ANorm;
println("|| P A - L U ||             = $decompError")
println("|| P A - L U || / || A ||   = $relDecompError")
# P A = L U implies inv(A) b = inv(P' L U) b = inv(U) inv(L) P b
x = U\(L\b[p]);
residError=norm(b-AOrig*x);
relResidError=residError/bNorm;
println("|| b - A x ||_2             = $residError")
println("|| b - A x ||_2 / || b ||_2 = $relResidError")

# Run the blocked code
println("Starting blocked partially-pivoted LU")
A = copy(AOrig);
tic();
p = LUPartial(A,96);
blockTime = toq();
println("Blocked time: $blockTime")
L = tril(A,-1) + Diagonal(ones(n));
U = triu(A);
decompError=norm(AOrig[p,:]-L*U);
relDecompError=decompError/ANorm;
println("|| P A - L U ||             = $decompError")
println("|| P A - L U || / || A ||   = $relDecompError")
# P A = L U implies inv(A) b = inv(P' L U) b = inv(U) inv(L) P b
x = U\(L\b[p]);
residError=norm(b-AOrig*x);
relResidError=residError/bNorm;
println("|| b - A x ||_2             = $residError")
println("|| b - A x ||_2 / || b ||_2 = $relResidError")


# In[4]:

function LURook(A)
    (m,n)=size(A);
    # LU with rook pivoting 
    p = [1:n];
    q = [1:n];
    totalPivCmps=0;
    totalRookMoves=0;
    for k=1:n
        # Search for an entry that is the maximum in its row and col 
        iPiv,jPiv,pivVal = k,k,abs(A[k,k]);
        rookMove=0;
        while true
            rookMove += 1;
            if rookMove % 2 == 1;
                iOld=iPiv;
                for i=k:n
                    if abs(A[i,jPiv]) > pivVal
                        iPiv,pivVal = i,abs(A[i,jPiv]);
                    end
                end
                iPiv==iOld && rookMove != 1 && break
            else
                jOld=jPiv;
                for j=k:n
                    if abs(A[iPiv,j]) > pivVal
                        jPiv,pivVal = j,abs(A[iPiv,j]);
                    end
                end
                jPiv==jOld && break
            end
        end
        totalPivCmps += rookMove*(n-k+1);
        totalRookMoves += rookMove;
    
        # Pivot the previous pieces of L and update p
        A[[k,iPiv],:] = A[[iPiv,k],:];
        p[[k,iPiv]  ] = p[[iPiv,k]  ];
    
        # Pivot U
        A[:,[k,jPiv]] = A[:,[jPiv,k]];
        q[  [k,jPiv]] = q[  [jPiv,k]];
    
        A[k+1:end,k] /= A[k,k];
        A[k+1:end,k+1:end] -= A[k+1:end,k]*A[k,k+1:end];
    end
    println("Total rook movements: $totalRookMoves")
    println("Total pivot comparisons: $totalPivCmps")
    return p, q
end

println("Starting rook-pivoted LU")
A = copy(AOrig);
tic();
p, q = LURook(A);
rookTime = toq();
println("Rook time: $rookTime")

# Expand A into L and U
L = tril(A,-1) + Diagonal(ones(n));
U = triu(A);
# Compute residuals for the original factorization and a linear solve
decompError=norm(AOrig[p,q] - L*U);
relDecompError=decompError/ANorm;
println("|| P A Q' - L U ||           = $decompError")
println("|| P A Q' - L U || / || A || = $relDecompError")
qInv=zeros(n);
for j=1:n
  qInv[q[j]] = j;
end
x = U\(L\b[p]);
x[:] = x[qInv];
residError=norm(b-AOrig*x);
relResidError=residError/bNorm;
println("|| b - A x ||_2             = $residError")
println("|| b - A x ||_2 / || b ||_2 = $relResidError")


# In[5]:


function LUFull(A)
    (m,n)=size(A);
    p = [1:n];
    q = [1:n];
    for k=1:n
        # Search for an entry that is the maximum in the remaining submatrix
        iPiv,jPiv,pivVal = k,k,abs(A[k,k]);
        for j=k:n
            for i=k:n
                if abs(A[i,j]) > pivVal
                    iPiv,jPiv,pivVal = i,j,abs(A[i,j]);
                end
            end
        end
    
        # Pivot the previous pieces of L and update p
        A[[k,iPiv],:] = A[[iPiv,k],:];
        p[[k,iPiv]  ] = p[[iPiv,k]  ];
    
        # Pivot U
        A[:,[k,jPiv]] = A[:,[jPiv,k]];
        q[  [k,jPiv]] = q[  [jPiv,k]];
    
        A[k+1:end,k] /= A[k,k];
        A[k+1:end,k+1:end] -= A[k+1:end,k]*A[k,k+1:end];
    end
    return p, q
end

println("Starting fully-pivoted LU")
A = copy(AOrig);
tic();
p, q = LUFull(A);
fullTime = toq();
println("Full time: $fullTime")
L = tril(A,-1) + Diagonal(ones(n));
U = triu(A);
# Compute and print the error in the decomposition and A \ b
decompError=norm(AOrig[p,q] - L*U);
relDecompError=decompError/ANorm;
println("|| P A Q' - L U ||           = $decompError")
println("|| P A Q' - L U || / || A || = $relDecompError")
# P A Q' = L U implies inv(A) b = inv(P' L U Q) b = Q' inv(U) inv(L) P b
qInv=zeros(n);
for j=1:n
  qInv[q[j]] = j;
end
x = U\(L\b[p]);
x[:] = x[qInv];
residError=norm(b-AOrig*x);
relResidError=residError/bNorm;
println("|| b - A x ||_2             = $residError")
println("|| b - A x ||_2 / || b ||_2 = $relResidError")

