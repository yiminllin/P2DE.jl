# Bisection. Assume f(x_valid) is satisfied,
#            find x in [x1,x2] closest to x2 so that f(x) is satisfied
function bisection(f,x_valid,x_invalid)
    if f(x_invalid)
        return x_invalid
    else
        maxit = 20
        iter  = 0
        while (iter <= maxit)
            x_new = .5*(x_valid+x_invalid)
            if f(x_new)
                x_valid = x_new
            else
                x_invalid = x_new
            end
            iter = iter+1
        end
        return x_valid
    end
end