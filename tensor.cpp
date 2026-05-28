#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

template<typename T>
class Tensor {
    private:
        std::vector<size_t> m_shape;
        std::vector<size_t> m_strides;
        std::shared_ptr<std::vector<T>> m_data;
    
    public:
        Tensor(std::vector<size_t> shape) 
        // can pass integer as m_strides, because it implicitly constructs a std::vector by just variable(arguments)
            : m_shape(std::move(shape)), m_strides(m_shape.size()) {
                m_strides[m_shape.size() - 1] = 1; 
                for (size_t i = m_shape.size() - 1; i > 0; --i) {
                    m_strides[i - 1] = m_shape[i] * m_strides[i];
                }
                m_data = std::make_shared<std::vector<T>>(m_shape[0] * m_strides[0]);
                // at first shared_ptr points to null. We gotta create it with a size.

            }

        Tensor(std::vector<size_t> shape, std::vector<size_t> strides, std::shared_ptr<std::vector<T>> data) // backdoor
            : m_shape(std::move(shape)), m_strides(std::move(strides)), m_data(std::move(data)) {} 

        Tensor(const Tensor& source) { // copy constructor (shallow copy)
            return Tensor(m_shape, m_strides, m_data); // boosts ref count by 1 (shallow copy)
        }

        // move constructor
        // copy assignment operator
        // move assignment operator

        Tensor clone() const {
            // *m_data is forwarded straight into vector construction zone. Sees a vector entering (*m_data is a vector) and triggers a totally new vector construction (new heap memory)
            auto ptr = std::make_shared<std::vector<T>>(*m_data); // shared_ptr pointing to a new vector with copied data. (ref_count = 1)
            return Tensor(m_shape, m_strides, std::move(ptr));
        }

        const std::vector<size_t>& shape() const {
            return m_shape;
        }

        const std::vector<size_t>& strides() const {
            return m_strides;
        }



};

template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vector) {
    for (size_t i = 0; i < vector.size(); ++i) {
        if (i == vector.size() - 1) {
            stream << vector[i];
        }
        else {
            stream << vector[i] << " ";
        }
        
    }
    return stream;
}

int main() {
    Tensor<float> tens(std::vector<size_t>{4, 3 ,2});
    std::cout << "Shape: [" << tens.shape() << "] | Strides: [" << tens.strides() << "]" << std::endl;
}