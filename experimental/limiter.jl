# TODO: refactor
function initialize_total_variation!(cache, prealloc, param, discrete_data, bcdata, dim::Dim1)
    (; equation) = param
    (; Uq) = prealloc
    (; mapP) = bcdata
    (; q2fq, fq2q) = discrete_data.ops
    (; s_modified, var_s_modified, lbound_s_modified) = cache

    N1D = param.N + 1
    K = num_elements(param)
    Nq = size(Uq, 1)
    Nfp = size(mapP, 1)
    # Preallocate total variation of s_modified at nodes: 
    # Section 4.7.1 https://arxiv.org/pdf/1710.00417.pdf 
    @batch for k = 1:K
        for i = 1:N1D
            stencil = low_order_stencil(i, k, N1D, Nfp, discrete_data, bcdata, dim)
            var_s_modified[i, k] = 2 * s_modified[i, k]
            for s in stencil
                s_modified_ij = s_modified[s...]
                var_s_modified[i, k] -= s_modified_ij
            end
        end
    end
end

# TODO: refactor
function initialize_total_variation!(cache, prealloc, param, discrete_data, bcdata, dim::Dim2)
    (; equation) = param
    (; Uq) = prealloc
    (; mapP) = bcdata
    (; q2fq, fq2q) = discrete_data.ops
    (; s_modified, var_s_modified, lbound_s_modified) = cache

    N1D = param.N + 1
    K = num_elements(param)
    Nq = size(Uq, 1)
    Nfp = size(mapP, 1)
    # Preallocate total variation of s_modified at nodes: 
    # Section 4.7.1 https://arxiv.org/pdf/1710.00417.pdf 
    @batch for k = 1:K
        s_modified_k = reshape(view(s_modified, :, k), N1D, N1D)
        var_s_modified_k = reshape(view(var_s_modified, :, k), N1D, N1D)
        for j = 1:N1D
            for i = 1:N1D
                stencil = low_order_stencil((i, j), k, N1D, Nfp, discrete_data, bcdata, dim)
                var_s_modified_k[i, j] = 4 * s_modified_k[i, j]
                for s in stencil
                    s_modified_ij = s_modified[s...]
                    var_s_modified_k[i, j] -= s_modified_ij
                end
            end
        end
    end
end

