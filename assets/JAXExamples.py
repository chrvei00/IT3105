# This file contains several simple examples using numpy's JAX package, which provides
# automatic differentiation for all kinds of functions, where many of the numpy primitives
# in each function have JAX equivalents.
# JAX is designed to follow principles of functional programming, so the differentiable
# functions should not include side effects: they should not do assignments to any
# non-local variables nor to any local variables whose values can persist from one
# call to the next.  In fact, JAX arrays (created via jax.numpy.array) CANNOT be
# modified in place.  Hence, anything that will be modified by functions that JAX
# traces should be included as an arguments to those functions.

import numpy as np
import matplotlib.pyplot as plt
import grapher as GR
import kd_array as KDA
import copy
import jax
import jax.numpy as jnp


# ****** Simple JAX test cases ******

def jaxf1(x,y):
    q = x**2 + 8
    z = q**3 + 5*x*y
    return z

def jaxf2(x,y):
    z = 1
    for i in range(int(y)):
        z *= (x+float(i))
    return z

def jaxf3(x,y):
    return x**y

df3a = jax.grad(jaxf3,argnums=0)
df3b = jax.grad(jaxf3,argnums=1)
df3c = jax.grad(jaxf3,argnums=[0,1])

def jaxf4(x,y):
    q = x**2 + 5
    r = q*y + x
    return q*r

df4 = jax.grad(jaxf4,argnums=[0,1])

def jaxf5(x,y):
    return jnp.array([x*y**3,x**3*y])

# Jax.jacrev => compute a Jacobian for reverse-mode autodiff.  We need a Jacobian, since jaxf5 outputs multiple
# values.
df5 = jax.jacrev(jaxf5,argnums=[0,1])

bad_news = 1

def jaxgum(x,y):
   global bad_news
   bad_news += 10
   return bad_news * x * y**2

# This does work, but it's bad practice, since dgum(1.0,1.0) gives a different value each time you call
# it.  Impure functions => loss of referential transparency.
dgum = jax.grad(jaxgum, argnums=[0,1])
dgum2 = jax.jit(dgum)  # A compiled version

def jaxhum(x,y,good_news):
    good_news += 10
    return good_news*x*y**2, good_news

# Two outputs, so we need a Jacobian for reverse-mode autodiff.
dhum = jax.jacrev(jaxhum,argnums=[0,1])
dhum2 = jax.jit(dhum)

# Testing out conditionals and iteration

def jumpinjax(x,n,switch,primes=[2,3,5,7,11]):
    switch = int(switch) # JAX Tracing requires real args, but range wants integers.
    if int(switch) == 0:
        for i in range(int(n)):
            x = x**2
    elif switch == 1:
        for p in primes:
            x = x*p
    else:   return - x
    return x

djuja = jax.grad(jumpinjax)

def jumpinjax2(x,n,switch,primes=[2,3,5,7,11]):
    n = int(n)  # JAX tracing requires reals, but
    switch = int(switch)
    if int(switch) == 0:    return ranger(x,n)
    elif switch == 1:   return primer(x,primes)
    else:   return - x
    return x

def ranger(y,m):
    for _ in range(int(m)):
        y = y**2
    return y

def primer(x,primes):
    for p in primes:
        x *= p
    return x

djuja2 = jax.grad(jumpinjax2)

def jumpinjax3(x,n,switch):
    if switch == 0: return ranger(x,n)
    elif switch == 1:
        return jnp.array([x**i for i in range(n)])
    else:   return -x

djuja3 = jax.jacrev(jumpinjax3)

''' ************  Tips for using JIT ************
In many cases, if you have a function like this:
def fff(x):
    if x > 0:   return x
    else:   return 0

fjit = jax.jit(fff)
fjit(33)  => error

JAX lets you compile fff, but at runtime, you'll probably get an error message.  The problem is that
at compile time, JIT does not know which branch of the
conditional will be taken, so it has trouble optimizing it.  The same problem arises if you try to replace
the conditional with a max:

def ggg(x):
    return max(x,0)

This is still a problem, since the python code for max includes a conditional.

The general problem is that whenever the control path in your code is dependent upon any of the function's arguments,
it is more difficult to optimize that code, so JIT returns some compiled code, but that code often fails at
 runtime. 

You can consult the JAX online documentation for some remedies / hacks.  Some of them are not worth the effort,
especially for these relatively small programming assignments where optimization is not that critical.

However, there is one solution for situations involving max and min:

    Just use jnp.maximum and jnp.minimum instead of the alternatives (max, numpy.max, jnp.max...) !!

def hhh(x):
    return jnp.maximum(x,0)

Now you should be able to do:
    hjit = jax.jit(hhh)

Calls to hjit should not cause any errors and should run very fast.

When writing code for dynamic systems, max and min pop up a lot, since you often need to clip values to be
within a particular range, such as [0, max-height] for modeling the height of water in a bathtub.
'''






