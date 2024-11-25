class BaseClass:
    def __init__(self, arg1, arg2, arg3):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
    
    def perform_operation(self):
        # Default implementation of the method
        return self.arg1 + self.arg2 + self.arg3


# Derived class that overrides the method and uses extra initialization arguments
class DerivedClass(BaseClass):
    def __init__(self, arg1, arg2, arg3, extra_arg):
        super().__init__(arg1, arg2, arg3)
        self.extra_arg = extra_arg

    def perform_operation(self):
        # Custom implementation using the extra argument
        return (self.arg1 + self.arg2 + self.arg3) * self.extra_arg


# Usage
base = BaseClass(1, 2, 3)
print(base.perform_operation())  # Output: 6

derived = DerivedClass(1, 2, 3, 10)
print(derived.perform_operation())  # Output: 60