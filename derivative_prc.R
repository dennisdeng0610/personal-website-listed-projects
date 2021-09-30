# coding=utf-8
# author: dennisdeng
# time: 07-12-2021 10:47:01


hermite <- function(x) {
  # use the first 4 Hermite orthogonal functions for the least square estimation
  #
  # arg x: the vector of stock prices at time t while EV > 0
  #
  # return: matrix L with length(x) rows and 4 columns, element (i,j) is L_j(x_i)
  L <- matrix(nrow = length(x), ncol = 4)
  L[, 1] <- 1 * x^0
  L[, 2] <- 2 * x
  L[, 3] <- 4 * x^2 - 2
  L[, 4] <- 8 * x^3 - 12 * x
  return(L)
}

lsmc_pricer <- function(sigma, r, K, s0, T, Npath) {
  # use lsmc method to price American put option
  #
  # arg sigma: volatility of the underlying stock
  # arg r: risk-free rate
  # arg K: strike price of the option
  # arg s0: current stock price
  # arg T: maturity time in years
  # arg Npath: number of simulated paths
  #
  # return: American put option price
  
  # set the time increment to be 1 / sqrt(Npath)
  # make sure the number of period is integer
  Nperiod <- floor(T / (1 / sqrt(Npath)))
  dt <- T / Nperiod
  
  # simulate the stock price in matrix
  # each row represents one independent path
  # each column represents one time period from 0 to T
  st <- matrix(nrow = Npath, ncol = Nperiod + 1)
  st[, 1] <- rep(s0, Npath) # set initial price equal to s0
  dwt <- matrix(
    data = rnorm(Npath * Nperiod / 2, sd = sqrt(dt)),
    nrow = Npath / 2, ncol = Nperiod
  )
  dwt <- rbind(dwt, -dwt) # use antithetic variate on half paths to reduce errors
  for (i in 1:Nperiod) {
    st[, i + 1] <- st[, i] + r * st[, i] * dt + sigma * st[, i] * dwt[, i]
  }
  
  # create 3 matrices for EV, index, and discount, with the same size of st
  # EV is the exercise value
  # index == 0 represents do not exercise, == 1 represents exercise
  # each column in the discount matrix represents the discount back factor
  ev <- K - st
  ev[ev < 0] <- 0 # if K < st, then exercise value = 0
  index <- matrix(data = 0, nrow = Npath, ncol = Nperiod + 1)
  index[which(ev[, Nperiod + 1] > 0), Nperiod + 1] <- 1
  discount <- matrix(
    data = rep(exp(-r * (seq(0, ncol(ev) - 1, 1)) * dt), Npath),
    nrow = Npath, ncol = Nperiod + 1, byrow = TRUE
  )
  
  # start backward from t_(n-1) to t_(2)
  for (i in Nperiod:2) {
    inmon <- which(ev[, i] > 0) # get path numbers where EV > 0 (in-the-money)
    x <- st[inmon, i] # x is the stock price at time i where EV(s_i)>0
    
    # get y values by past EV values, 0/1 index, and discount factors
    if (i == Nperiod) {
      y <- discount[inmon, 2:(Nperiod - i + 2)] *
        index[inmon, (i + 1):(Nperiod + 1)] * ev[inmon, (i + 1):(Nperiod + 1)]
    } else {
      y <- rowSums(discount[inmon, 2:(Nperiod - i + 2)] *
                     index[inmon, (i + 1):(Nperiod + 1)] * ev[inmon, (i + 1):(Nperiod + 1)])
    }
    
    # get A,b,a
    A <- t(hermite(x)) %*% hermite(x)
    b <- t(hermite(x)) %*% y
    a <- chol2inv(chol(A)) %*% b # a = A^(-1)b, use cholesky root
    
    # get ECV and compare it with ev, if EV>ECV, then exercise at time i,
    # all other index from the same row = 0 (each path at most exercise once)
    ecv <- hermite(x) %*% a
    compare <- ev[inmon, i] - ecv
    index[inmon[which(compare > 0)], ] <- 0
    index[inmon[which(compare > 0)], i] <- 1
  }
  
  # get the output by sum all the values and take the average by divide the number of paths
  output <- sum(index * ev * discount) / Npath
  return(output)
}

pde_pricer <- function(sigma, r, K, N, T, delta_t, delta_s, alpha) {
  # use PDE method to price American put option
  #
  # arg sigma: volatility of the underlying stock
  # arg r: risk-free rate
  # arg K: strike price of the option
  # arg N: (N+1) number of initial prices
  # arg T: maturity time in years
  # arg delta_t: time increment
  # arg delta_s: price increment
  # arg alpha: determine the type of finite difference, alpha=0.5 is Crank-Nicolson
  #
  # return: American put option prices with difference initial prices
  
  # set the price range and time range
  s_range <- seq(delta_s * N, 0, -delta_s)
  t_range <- seq(0, T, delta_t)
  
  # set the grid of option prices
  grid <- matrix(nrow = length(s_range), ncol = length(t_range))
  prc_index <- seq(N - 1, 1, -1)
  terminal <- pmax(K - s_range, 0)
  grid[, ncol(grid)] <- terminal # set the terminal prices
  
  # get matrix A, for j = N-1,N-2,...,1
  getA <- function(j) {
    a1 <- (sigma^2 * j^2 - r * j) * (1 - alpha) * 0.5
    a2 <- -1 / delta_t - (sigma^2 * j^2 + r) * (1 - alpha)
    a3 <- (sigma^2 * j^2 + r * j) * (1 - alpha) * 0.5
    zero <- matrix(data = 0, nrow = (N - 1), ncol = (N - 1))
    target <- as.vector(t(cbind(a3, a2, a1, zero)))
    A <- matrix(data = target[1:(length(target) - (N - 1))], byrow = TRUE, nrow = (N - 1))
    A <- A[, 2:(ncol(A) - 1)]
    return(A)
  }
  outputA <- getA(prc_index)
  
  # get b1,b2,b3 values, for j = N-1,N-2,...,1
  getB <- function(j) {
    b1 <- (sigma^2 * j^2 - r * j) * alpha * 0.5
    b2 <- 1 / delta_t - (sigma^2 * j^2 + r) * alpha
    b3 <- (sigma^2 * j^2 + r * j) * alpha * 0.5
    return(cbind(b1, b2, b3))
  }
  B <- getB(prc_index)
  
  # recursion from time T to 0
  for (time in ncol(grid):2) {
    outputB <- c()
    for (i in 1:(nrow(grid) - 2)) {
      outputB[i] <- sum(-B[i, ] * grid[(i + 2):i, time]) # get vector B
    }
    price_mid <- solve(outputA, outputB) # C^i = A^-1 %*% B^(i+1)
    
    # extreme cases for put options
    price_top <- price_mid[1]
    price_bottom <- price_mid[length(price_mid)] + delta_s
    
    grid[, time - 1] <- c(price_top, price_mid, price_bottom)
    grid[, time - 1] <- pmax(grid[, time - 1], terminal) # american type
  }
  return(grid[, 1])
}


# compare between s0=40,...,36
answerA <- c()
for (i in 40:36) {
  answerA <- append(answerA,lsmc_pricer(0.2,0.06,40,i,1,100000))
}
answerB <- pde_pricer(0.2, 0.06, 40, 60, 1, 0.002, 1, 0.5)
answerB <- answerB[21:25]
result <- matrix(data=c(answerA,answerB),nrow=2,byrow = TRUE)
rownames(result) <- c('LSMC','PDE')
colnames(result) <- c('s0=40','s0=39','s0=38','s0=37','s0=36')
