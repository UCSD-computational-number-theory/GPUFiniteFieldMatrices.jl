pluq_init_perm_ka(n::Int) = pluq_init_perm(n)
pluq_inverse_perm_ka(p::Vector{Int}) = pluq_inverse_perm(p)
pluq_compose_segment_ka!(perm::Vector{Int}, offset::Int, locperm::Vector{Int}) = pluq_compose_segment!(perm, offset, locperm)
