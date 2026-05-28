#include <memory>

template <typename T>
class Node {
    public:
        virtual T evaluate() const = 0;

        virtual ~Node() {
            //std::cout << "Node destructor called" << std::endl;
        }
};

template <typename T>
class ValueNode : public Node<T> { // ValueNode IS Node
    private:
        T data;

    public:
        ValueNode(T value) : data(std::move(value)) {}
        
        T evaluate() const override {return data;}

        void set_value(T value) { // pass by value
            data = std::move(value);
        }

};

template <typename T>
class AddNode : public Node<T> {
    private:
        std::shared_ptr<Node<T>> m_left;
        std::shared_ptr<Node<T>> m_right;
    public:
        AddNode(std::shared_ptr<Node<T>> left, std::shared_ptr<Node<T>> right) // passing by value increments their reference counter
            : m_left(std::move(left)), m_right(std::move(right)) {} // without std::move it gets COPIED. Primitive types dont have move semantics,
            // but std::shared_ptr and others do have. We want to avoid creating a copy to then destroy it once curly brace "}" hits.
            // We want to copy by value (increment refs) and move it to the correct place.

        T evaluate() const override  {
            return m_left->evaluate() + m_right->evaluate();
        }
};

template <typename T>
class MultiplyNode : public Node<T> {
    private: 
        std::shared_ptr<Node<T>> m_left;
        std::shared_ptr<Node<T>> m_right;
    public:
        MultiplyNode(std::shared_ptr<Node<T>> left, std::shared_ptr<Node<T>> right)
            : m_left(std::move(left)), m_right(std::move(right)) {}

        T evaluate() const override {
            return m_left->evaluate() * m_right->evaluate();
        }
};

// function pointer looks like this: ReturnType (*PointerName)(Parameter Types);
template <typename T>
class LambdaNode : public Node<T> {
    private:
        std::shared_ptr<Node<T>> m_value;
        T (*m_func)(T); // define a var named m_func that is a function pointer.
    public:
        LambdaNode(std::shared_ptr<Node<T>> value, T (*func)(T)) // a function pointer
            : m_value(std::move(value)), m_func(func) {} // function pointer is primitive
        
        T evaluate() const override {
            return m_func(m_value->evaluate());
        }
};

template <typename T>
std::shared_ptr<Node<T>> operator+(std::shared_ptr<Node<T>> lhs, std::shared_ptr<Node<T>> rhs) {
    return std::make_shared<AddNode<T>>(std::move(lhs), std::move(rhs));
} // golden rule: If a Child class publicly inherits from Base class, a pointer to the Child can be implicitly converted
// into a pointer to the Base, anytime, anywhere, for free

template <typename T>
std::shared_ptr<Node<T>> operator*(std::shared_ptr<Node<T>> lhs, std::shared_ptr<Node<T>> rhs) {
    return std::make_shared<MultiplyNode<T>>(std::move(lhs), std::move(rhs));
}

template <typename T>
std::shared_ptr<Node<T>> operator+(std::shared_ptr<Node<T>> lhs, const T rhs) {
    return std::make_shared<AddNode<T>>(std::move(lhs), std::make_shared<ValueNode<T>>(rhs)); // the make_shared as argument will be MOVED bc its an Rvalue
}

template <typename T>
std::shared_ptr<Node<T>> operator+(const T lhs, std::shared_ptr<Node<T>> rhs) {
    return std::make_shared<AddNode<T>>(std::make_shared<ValueNode<T>>(lhs), std::move(rhs)); // the make_shared as argument will be MOVED bc its an Rvalue
}

template <typename T>
std::shared_ptr<Node<T>> operator*(std::shared_ptr<Node<T>> lhs, const T rhs) {
    return std::make_shared<MultiplyNode<T>>(std::move(lhs), std::make_shared<ValueNode<T>>(rhs)); // the make_shared as argument will be MOVED bc its an Rvalue
}

template <typename T>
std::shared_ptr<Node<T>> operator*(const T lhs, std::shared_ptr<Node<T>> rhs) {
    return std::make_shared<MultiplyNode<T>>(std::make_shared<ValueNode<T>>(lhs), std::move(rhs)); // the make_shared as argument will be MOVED bc its an Rvalue
}

// TEST SUITE WRITTEN BY A CLANKER
int main() {


    return 0;
}