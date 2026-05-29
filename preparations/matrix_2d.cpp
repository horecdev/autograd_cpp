#include <iostream>
#include <ostream>

struct Matrix2D {
    public:
        // CONSTRUCTORS AND DESTRUCTOR
        Matrix2D()
            : m_rows(0), m_cols(0), m_data(nullptr) {++matrix_allocations;}

        Matrix2D(const int rows, const int cols)
            : m_rows(rows), m_cols(cols), m_data(new float[rows * cols]()) {++matrix_allocations;}

        Matrix2D(const int rows, const int cols, const float* source)
            : m_rows(rows), m_cols(cols) {
                int size = m_rows * m_cols;
                m_data = new float[size];
                for (int i = 0; i < size; ++i) {
                    m_data[i] = source[i];
                }
                ++matrix_allocations;
            }

        ~Matrix2D() {
            delete[] m_data;
            --matrix_allocations;
        }

        // RULE OF 5

        Matrix2D(const Matrix2D& source) // copy constructor | turns M2D m1 = m2 into M2D m1(m2)
            : m_rows(source.m_rows), m_cols(source.m_cols), m_data(new float[m_rows * m_cols]) {
                for (int i = 0; i < m_rows * m_cols; ++i) {
                    m_data[i] = source.m_data[i];
                }
                ++matrix_allocations;
            }

        Matrix2D(Matrix2D&& source) noexcept // move constructor | M2D m1 = m2 into M2D m1(m2) where m2 is && Rvalue
            : m_rows(source.m_rows), m_cols(source.m_cols) {
                m_data = source.m_data;
                source.m_rows = 0;
                source.m_cols = 0;
                source.m_data = nullptr;

                ++matrix_allocations;
            }

        Matrix2D& operator=(const Matrix2D& other) { // copy assignment | m3 = m4 turns into m3.operator=(m4) and m4 is just &
            if (this != &other) {
                delete[] m_data;
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_data = new float[m_rows * m_cols];
                for (int i = 0; i < m_rows * m_cols; ++i) {
                    m_data[i] = other.m_data[i];
                }
            }
            return *this; // reference to new object
        }
        

        Matrix2D& operator=(Matrix2D&& other) noexcept { // move assignment | same but m4 is Rvalue &&
            if (this != &other) { // otherwise its just a REFERENCE dying so we dont have to worry. It doesnt call the destructor
                delete[] m_data;
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_data = other.m_data;

                other.m_rows = 0;
                other.m_cols = 0;
                other.m_data = nullptr;
            }

            return *this;
        }

        // OPERATORS AND MATH
        
        Matrix2D& operator+=(const Matrix2D& rhs) {
            for (int i = 0; i < m_rows * m_cols; ++i) {
                m_data[i] += rhs.m_data[i];
            }
            return *this;
        }

        friend Matrix2D operator+(Matrix2D lhs, const Matrix2D& rhs) { // lhs is passed BY VALUE making a COPY
            lhs += rhs;
            return lhs;
        }

        Matrix2D& operator+=(const float scalar) {
            for (int i = 0; i < m_rows * m_cols; ++i) {
                m_data[i] += scalar;
            }
            return *this;
        }

        friend Matrix2D operator+(Matrix2D lhs, float scalar) {
            lhs += scalar;
            return lhs; // returns an rvalue
        }

        friend Matrix2D operator+(const float scalar, Matrix2D rhs) {
            return std::move(rhs) + scalar; // 1. Copies rhs | 2. turns it into Rvalue | 3. passes into func above with Move Constructor, not Copy. | 4. does math
            // it avoids unnecessary allocations
            // could just do rhs += scalar;
        }

        Matrix2D& operator*=(const Matrix2D& rhs) {
            for (int i = 0; i < m_rows * m_cols; ++i) {
                m_data[i] *= rhs.m_data[i];
            }
            return *this;
        }

        friend Matrix2D operator*(Matrix2D lhs, const Matrix2D& rhs) {
            lhs *= rhs;
            return lhs; // returns an Rvalue which then triggers Move actions
        }

        Matrix2D& operator*=(const float scalar) {
            for (int i = 0; i < m_rows * m_cols; ++i) {
                m_data[i] *= scalar;
            }
            return *this;
        }

        friend Matrix2D operator*(Matrix2D lhs, const float scalar) {
            lhs *= scalar;
            return lhs;
        }

        friend Matrix2D operator*(const float scalar, Matrix2D rhs) {
            rhs *= scalar;
            return rhs;
        }

        // WHEN YOU DO RESULT = (A + B) * 2.0f: 
        // | 1. (A + B) does 1 copy and is stored in an Rvalue 
        // | 2. Triggers Rvalue * scalar, instead of copying it, it uses the Move Constructor to push Rvalue into LHS
        // | 3. Rvalue * 2.0f results in an Rvalue (evaluated in place) 
        // | 4. Move Assignment Operator moves it into result variable.

        Matrix2D matmul(const Matrix2D& rhs) const {
            const Matrix2D& self = *this;

            Matrix2D result = Matrix2D(self.get_rows(), rhs.get_cols());
            for (int r = 0; r < self.get_rows(); ++r) {
                for (int c = 0; c < rhs.get_cols(); ++c) {
                    float current_res = 0.0f; // put into [r, c]
                    for (int mul_idx = 0; mul_idx < m_cols; ++mul_idx) {
                        current_res += self[r, mul_idx] * rhs[mul_idx, c];
                    }
                    result[r, c] = current_res;
                }
            }
            return result;
        }

        Matrix2D transpose() const {
            const Matrix2D& self = *this;

            Matrix2D result = Matrix2D(m_cols, m_rows);
            // [r, c] of source goes into [c, r] of result
            for (int r = 0; r < m_rows; ++r) {
                for (int c = 0; c < m_cols; ++c) {
                    result[c, r] = self[r, c];
                }
            }
            return result;
        }
        
        float& operator[](const int row, const int col) {
            return m_data[row * m_cols + col];
        }

        float operator[](const int row, const int col) const {
            return m_data[row * m_cols + col];
        }

        // GETTERS / UTILS

        int get_rows() const {return m_rows;}
        int get_cols() const {return m_cols;}
        static int get_allocations() {return matrix_allocations;}

        friend std::ostream& operator<<(std::ostream& stream, const Matrix2D& mat) { // it is a function OUTSIDE struct scope, but we just put it inside so its pretty.
            stream << "Shape: (" << mat.m_rows << "x" << mat.m_cols << ")" << std::endl;
            for (int r_idx = 0; r_idx < mat.m_rows; ++r_idx) {
                for (int c_idx = 0; c_idx < mat.m_cols; ++c_idx) {
                    std::cout << mat.m_data[r_idx * mat.m_cols + c_idx] << " ";
                }
                std::cout << std::endl;
            }
            return stream;
        }



    private:
        int m_rows;
        int m_cols;
        float* m_data;

        static int matrix_allocations;
};

int Matrix2D::matrix_allocations = 0;

int main() {
    return 0;
}