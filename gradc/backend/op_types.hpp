enum class UnaryOp {Identity, ReLU, Sigmoid, Exp, Log};
enum class UnaryOpInPlace {ReLU, Sigmoid, Exp, Log};
enum class BinaryOp {Add, Sub, Mul, Div, MatMul};
enum class BinaryOpInPlace {Add, Sub, Mul, Div};
enum class ReduceOp {Sum, Mean, Max, Min};
enum class LayoutOp {Concat, Stack};