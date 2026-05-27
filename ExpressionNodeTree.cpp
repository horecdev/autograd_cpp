#include <iostream>
#include <memory>
#include <functional>

template <typename T>
class Node {
    virtual T evaluate() const = 0;

    virtual ~Node() {
        std::cout << "Node destructor called" << std::endl;
    }
};

template <typename T>
class ValueNode : public Node<T> {
    private:
        T data;

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
