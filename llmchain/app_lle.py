class Runnable:
    def __init__(self, func):
        # Save the function we want this Runnable to wrap
        self.func = func  

    def __or__(self, other):
        """
        This allows us to use the "|" operator (pipe) to chain Runnables.
        Example: a | b  means "run a, then pass the result into b".
        """
        def chained_func(*args, **kwargs):
            # Call the first function (self.func) with any arguments
            result = self.func(*args, **kwargs)

            # Pass the result into the next Runnable (other)
            # NOTE: other must also be a Runnable with an invoke() method
            return other.invoke(result)

        # Return a new Runnable that represents the chained functions
        return Runnable(chained_func)

    def invoke(self, *args, **kwargs):
        """
        Execute the wrapped function.
        *args = any number of positional arguments (tuple)
        **kwargs = any number of keyword arguments (dict)
        """
        return self.func(*args, **kwargs)


a = Runnable(lambda x: x * 2)   # wraps a "double" function
b = Runnable(lambda x: x + 3)   # wraps an "add 3" function

# Thanks to __or__, we can do this:
pipeline = a | b

print(pipeline.invoke(5))  
# Step 1: a(5) = 10
# Step 2: b(10) = 13
# Output: 13
