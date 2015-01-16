# In[1]:

n = 1000;
# Build a random SPD matrix
B = randn(n,n);
AOrig = B*B';
ANorm = norm(AOrig);
println("|| A ||_2 = $ANorm")
ACond = cond(AOrig);
println("cond(A) = $ACond")
# Build a random right-hand sidel
b = randn(n);
bNorm = norm(b);
println("|| b ||_2 = $bNorm")


# In[2]:

function CholUnb(A)
    m,n = size(A);
    # Unblocked Cholesky
    for k=1:n
        A[k,k] = sqrt(A[k,k]);
        A[k+1:end,k] /= A[k,k];
        # NOTE: There is no 'syr'/'her' in Julia. Pull request?
        A[k+1:end,k+1:end] -= A[k+1:end,k]*A[k+1:end,k]';
    end
end

function Chol(A,bsize)
    m,n = size(A);
    # Blocked Cholesky
    for k=1:bsize:n
        nb = min(n-k+1,bsize);
        ind1 = k:k+nb-1;
        ind2 = k+nb:n;
        A11 = sub(A,ind1,ind1);
        A21 = sub(A,ind2,ind1);
        A22 = sub(A,ind2,ind2);
        CholUnb(A11);
        BLAS.trsm!('R','L','C','N',1.,A11,A21);
        # NOTE: 'herk' does not fall through to 'syrk' for real matrices.
        #        Pull request?
        BLAS.syrk!('L','N',-1.,A21,1.,A22);
    end
end

# A warmup round for cache consistency in the timings
A = copy(AOrig);
Chol(A,96);

# Run and time the unblocked algorithm
A = copy(AOrig);
tic();
CholUnb(A);
unbTime=toq();
println("unblocked time: $unbTime")

# Run and time the blocked algorithm
A = copy(AOrig);
tic();
Chol(A,96);
blockTime=toq();
println("blocked time: $blockTime")

# Check the error of the blocked algorithm
L = tril(A);
decompError=norm(AOrig - L*L');
relDecompError=decompError/ANorm;
println("|| A - L L' ||             = $decompError")
println("|| A - L L' || / || A ||   = $relDecompError")
# A = L L' implies inv(A) b = inv(L L') b = inv(L') inv(L) b
x = L'\(L\b);
residError=norm(b-AOrig*x);
relResidError=residError/bNorm;
println("|| b - A x ||_2             = $residError")
println("|| b - A x ||_2 / || b ||_2 = $relResidError")

# Check the error with the built-in solve
tic();
x = AOrig\b;
builtinTime=toq();
println("built-in solve time: $builtinTime")
residError=norm(b-AOrig*x);
relResidError=residError/bNorm;
println("|| b - A x ||_2             = $residError (backslash)")
println("|| b - A x ||_2 / || b ||_2 = $relResidError (backslash)")

