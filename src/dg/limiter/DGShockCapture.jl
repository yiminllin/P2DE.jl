################################
###   Smoothness indicator   ###
################################
function initialize_smoothness_indicator!(shockcapture_type::NoShockCapture, bound_type::PositivityBound, prealloc, param, discrete_data, nstage)
    # Do nothing
end

# Initialize smoothness indicator when using subcell limiter w/ min entropy bound
#                              or when using modal shock capture scheme
function initialize_smoothness_indicator!(shockcapture_type, bound_type, prealloc, param, discrete_data, nstage)
    dim = dim_type(param.equation)
    initialize_smoothness_indicator!(prealloc, param, discrete_data, nstage, dim)
end

function initialize_smoothness_indicator!(prealloc, param, discrete_data, nstage, dim::Dim1)
    (; indicator, indicator_modal, smooth_indicator) = prealloc
    (; N, equation) = param
    (; VDM_inv) = discrete_data.ops
    (; K) = discrete_data.sizes

    initialize_indicator!(equation, prealloc, param, discrete_data, nstage)

    @batch for k = 1:K
        @views mul!(indicator_modal[:, k], VDM_inv, indicator[:, k])
    end

    # TODO: hardcoded for quad
    @batch for k = 1:K
        count = 1
        modeN_energy = 0.0
        modeNm1_energy = 0.0
        total_energy = 0.0
        for i = 0:N
            energy = indicator_modal[count, k]^2
            if i == N
                modeN_energy += energy
            end
            if i == N - 1
                modeNm1_energy += energy
            end
            total_energy += energy
            count += 1
        end
        smooth_indicator[k] = max(modeN_energy / total_energy, modeNm1_energy / total_energy)
    end
end

function initialize_smoothness_indicator!(prealloc, param, discrete_data, nstage, dim::Dim2)
    (; indicator, indicator_modal, smooth_indicator) = prealloc
    (; N, equation) = param
    (; VDM_inv) = discrete_data.ops
    (; K) = discrete_data.sizes

    initialize_indicator!(equation, prealloc, param, discrete_data, nstage)

    @batch for k = 1:K
        @views mul!(indicator_modal[:, k], VDM_inv, indicator[:, k])
    end

    # TODO: hardcoded for quad
    @batch for k = 1:K
        count = 1
        modeN_energy = 0.0
        modeNm1_energy = 0.0
        total_energy = 0.0
        for j = 0:N
            for i = 0:N
                energy = indicator_modal[count, k]^2
                if i == N || j == N
                    modeN_energy += energy
                end
                if i == N - 1 || j == N - 1
                    modeNm1_energy += energy
                end
                total_energy += energy
                count += 1
            end
        end
        smooth_indicator[k] = max(modeN_energy / total_energy, modeNm1_energy / total_energy)
    end
end

function initialize_indicator!(equation::CompressibleIdealGas, prealloc, param, discrete_data, nstage)
    (; indicator) = prealloc
    (; equation) = param
    (; Uq) = prealloc
    (; K, Nq) = discrete_data.sizes

    @batch for k = 1:K
        for i = 1:Nq
            Ui = Uq[i, k]
            rho = Ui[1]
            p = pfun(equation, Ui)
            indicator[i, k] = rho * p
        end
    end
end

function initialize_indicator!(equation::KPP, prealloc, param, discrete_data, nstage)
    (; indicator) = prealloc
    (; equation) = param
    (; Uq) = prealloc
    (; K, Nq) = discrete_data.sizes

    @batch for k = 1:K
        for i = 1:Nq
            Ui = Uq[i, k][1]
            indicator[i, k] = Ui
        end
    end
end

#############################
### shock blending factor ###
#############################
function update_blending_factor!(shockcapture::NoShockCapture, cache, prealloc, param, discrete_data, nstage)
    # No blending by default
    @views @. cache.blending_factor[:, nstage] = 1.0
end

# (46) in https://www.sciencedirect.com/science/article/pii/S0021999120307099
function update_blending_factor!(shockcapture::HennemannShockCapture, cache, prealloc, param, discrete_data, nstage)
    (; blending_factor) = cache
    (; smooth_indicator) = prealloc
    (; a, c) = shockcapture
    (; N) = param
    (; K) = discrete_data.sizes

    TN = a * 10^(-c * (N + 1)^0.25)
    alphamax = 0.5
    alphaE0 = 0.0001
    s_factor = log((1 - alphaE0) / alphaE0)
    @batch for k = 1:K
        alpha = 1.0 / (1.0 + exp(-s_factor / TN * (smooth_indicator[k] - TN)))
        blending_factor[k, nstage] = max(min(1.0 - alpha, 1.0), alphamax)
    end
end
