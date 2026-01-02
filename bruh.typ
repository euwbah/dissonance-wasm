

#let cont_frac(x, iters: 10) = {
  let leftover = x - calc.round(x)
  let b-n = (calc.floor(1/leftover),)
  for i in range(iters) {
    let x-n = 1/leftover
    leftover = x-n - calc.floor(x-n)
    b-n.push(calc.floor(1/leftover))
  }
  b-n
}


#cont_frac(calc.pi)