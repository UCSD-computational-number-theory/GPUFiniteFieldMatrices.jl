struct KMat{MatrixType,Int}
    l::MatrixType
    h::MatrixType
    m::Int
end

struct KMatMulPlan{MatrixType}
    temp1::MatrixType
    temp2::MatrixType
end

function KMatMul!(C,A,B,plan)
    plan.temp1 = A.l + A.h
    plan.temp2 = B.l + B.h
    C.l = A.l*B.l
    C.h = plan.temp1*plan.temp2
    C.h = C.h - C.l
    C.h = C.h - A.h*B.h
end

function KMatToMat(K)
    A = zeros(eltyple(K.l),nrows(K.l),ncols(K.l))
    A = K.l + K.m*K.h
    return A
end

function MatToKMat(A,M)
    K = KMat{Matrix{eltype(A)},Int}(zeros(eltype(A),nrows(A),ncols(A)),zeros(eltype(A),nrows(A),ncols(A)),M)
    K.h = trunc.(A./M)
    K.l = A - K.h
    K.m = M
    return K
end