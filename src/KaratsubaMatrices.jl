module KaratsubaMatrices

struct KaratsubaMatrix{MatrixType}
    l::MatrixType
    h::Matrixtype
    m::Int
end

struct KaratsubaMatMulPlan{MatrixType}
    temp1::MatrixType
    temp2::MatrixType
end

function KaratsubaMatMul!(C,A,B,plan)
    plan.temp1 = A.l + A.h
    plan.temp2 = B.l + B.h
    C.l = A.l*B.l
    C.h = plan.temp1*plan.temp2
    C.h = C.h - C.l
    C.h = C.h - A.h*B.h
end
end