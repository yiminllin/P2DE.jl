# TODO: not working now...
# https://math.stackexchange.com/questions/3551775/bracketing-root-finding-methods-my-modified-illinois-method 
function regula_falsi(f,x_valid,x_invalid)
    if f(x_invalid) > 0
        return x_invalid
    else
        maxit = 10
        iter  = 0
        tol   = 1e-10
        a = x_valid
        b = x_invalid
        fa = f(a)
        fb = f(b)
        acopy  = a
        facopy = fa
        if abs(fa) > abs(fb)
            a  = b
            fa = fb
            b  = acopy
            fa = facopy
        end
        while iter <= maxit && abs(a-b) > tol
            c = a - (fa*(b-a))/(fb-fa)
            fc = f(c)
            if fc*fa <= 0
                b  = a
                fb = fa
            else
                fb = .5*fb
            end
            a  = c
            fa = fc
            iter = iter+1
        end
        return fa >= 0 ? a : b
    end
end

# Secant method. Assume f(xl) > 0
#            find x in [xl,xr] closest to xr so that f(x) > 0
#            TODO: Assume f is a convex constraint
function newton_secant(f,df,xl,xr)
    maxit = 10
    iter  = 0
    tol   = 1e-6
    if f(xr) > 0
        return xr
    end
    # Invariant: fl > 0, fr < 0
    while iter <= maxit && xr-xl > tol
        iter = iter + 1
        xprev = xl
        # Secant solve from the left
        fl = f(xl)
        fr = f(xr)
        if fl > fr
            sl = (fr-fl)/(xr-xl)
            xl = xl - fl/sl
        else
            break
        end
        try 
            fl = f(xl)
            # Sanity check, if we should break, backtrack and break
            if xl > xr || fl < 0
                xl = xprev
                break
            end
        catch err
            if isa(err, DomainError)
                xl = xprev
                break
            else
                throw(err)
            end
        end
        # Newton from the right
        dfr = df(xr)
        if dfr < 0
            xr = xr - fr/dfr
        else
            break
        end
        # Sanity check
        if f(xr) > 0
            xl = xr
            break
        end
    end
    return xl
end

@inline function s_modified_grad_ufun(equation::CompressibleIdealGas,U)
    rho,rhou,rhov,E = U
    γ = get_γ(equation)
    rhoe = rhoe_ufun(equation,U)
    rhomγ = rho^(-γ-1)
    g1 = -E*rhomγ + .5*(1+γ)*rhomγ/rho*(rhou^2+rhov^2)
    g2 = -rhou*rhomγ
    g3 = -rhov*rhomγ
    g4 = rho*rhomγ
    return SVector(g1,g2,g3,g4)
end

