#!/usr/bin/env python
import tensorflow as tf

print(" = Section 1 = ")
with tf.variable_scope("foo") as foo_scope:
    print("foo.reuse: ", foo_scope.reuse) # reuse = False
    v = tf.get_variable("v", [1])  # foo/v:0;  0 means the first output from the op
    print ("v.name: ", v.name)
print()

print(" = Section 2 = ")
with tf.variable_scope(foo_scope):  # reuse same scope, but don't set reuse=True
    print("foo.reuse: ", foo_scope.reuse)  # reuse = False by default!
    #v = tf.get_variable("v", [1])   # foo/v:0   CANNOT retrieve the old var under old scope. False enforces novelty.
    #print("v.name: ", v.name)
    w = tf.get_variable("w", [1])   # foo/w:0   Can ONLY do new variables under old scope with reuse=False.
    print("w.name: ", w.name)
print()

print(" = Section 3 = ")
with tf.variable_scope("outer_scope") as outer_scope:
    assert (outer_scope.name == "outer_scope")
    print("Outer scope: " + str(outer_scope.name))
print()

print(" = Section 4 = ")
p = tf.get_variable("p", [1])  # just p, no scope
print("p.name: ", p.name)
print()

print(" = Section 5 = ")
# will fail if reuse=False (=default), since the vars are already declared above
with tf.variable_scope(foo_scope, reuse=True) as foo_scope2:
    print("foo_scope2: ", foo_scope2.reuse)
    # Once reuse=True, all inner scoping MUST be true as well.
    v1 = tf.get_variable("v", [1])  # foo/v:0
    print("v1.name: ", v1.name)
    w1 = tf.get_variable("w", [1])  # foo/w:0
    print("w1.name: ", w1.name)
    #y1 = tf.get_variable("y", [1])  # foo/y:0  # Under reuse=True, can't do new variables.
    #print("y1.name: ", y1.name)
print()

print(" = Section 6 = ")
with tf.variable_scope("some_random_scope") as random_scope:
    with tf.variable_scope("some_new_random_scope") as new_random_scope:
        w2 = tf.get_variable("w2", [1])  # some_random_scope/some_new_random_scope/w2:0
        print("w2.name: ", w2.name)
        with tf.variable_scope(outer_scope) as new_outer_scope:  # will 'pop out' of the previous scopes.
            w3 = tf.get_variable("w", [1])  # outer_scope/w:0, from above
            print("w3.name: ", w3.name)
            with tf.variable_scope(foo_scope) as new_foo_scope:  # will 'pop out' of the previous scopes.
                z = tf.get_variable("z", [1]) # foo/z:0
                print("z.name: ", z.name)
print()

assert v1 == v
assert w1 == w

print(" = Section 7 = ")
# Will fail with reuse=False
try:
    with tf.variable_scope(foo_scope, reuse=False):
        w2 = tf.get_variable("w", [1]) # value error; we already defined "w" above, can only define new vars under reuse=False
        print("Should not get here.")
except ValueError:
    print("Correctly threw value error; can't define an old var under reuse=False.")
print()

print(" = Section 8 = ")
# Will fail with reuse=True
try:
    with tf.variable_scope(foo_scope, reuse=True):
        n1 = tf.get_variable("n", [1]) # value error; n is a new variable, but reuse is True.
        print("Should not get here.")
except ValueError:
    print("Correctly threw value error; can't define a new var under reuse=True.")


# See tf.AUTO_REUSE as well
