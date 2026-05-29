#include <iostream>

constexpr int compute_safe_padding(int size) {
    return ((15 + size) / 16) * 16;
}

int MAX_BUFFER_SIZE = compute_safe_padding(63); // 64

struct MyString {
    private:
        static int instances_alive;
        static int logs_printed;

        int m_size;
        char* m_string;

    public:
        friend std::ostream& operator<<(std::ostream& stream, const MyString& str); // global function, does not have "this"

    ~MyString() {
        delete[] m_string;
        instances_alive--;
    }

    MyString() 
        : m_size(0), m_string(new char[1]()) {instances_alive++;} // [1]() automatically wipes to '\0' at first byte

    MyString(const char* source) // copy from a const char* array
        {
            int l = 0;
            while (source[l] != '\0') {l++;}

            m_size = l;
            m_string = new char[m_size+1];

            for (int i = 0; i < l; i++) {
                m_string[i] = source[i];
            }
            m_string[m_size] = '\0'; // set last elem to be null
            instances_alive++;
        }

    explicit MyString(const int size)
        : m_size(size), m_string(new char[size + 1]()) {instances_alive++;}

    MyString(const char letter, const int size) 
        : m_size(size), m_string(new char[size + 1]) {
            for (int i = 0; i < size; i++) {
                m_string[i] = letter;
            }
            m_string[size] = '\0';
            instances_alive++;
        }
    
    MyString(const MyString& other) // COPY CONSTRUCTOR: MyString c = d BORN FROM an LVALUE
        : m_size(other.m_size), m_string(new char[other.m_size + 1]) {
            for (int i = 0; i < m_size; i++) {
                m_string[i] = other.m_string[i];
            }
            m_string[m_size] = '\0';
            instances_alive++;
        }

    MyString(MyString&& other) noexcept // MOVE CONSTRUCTOR: MyString var = function_that_returns_MyString() BORN FROM RVALUE
        : m_size(other.m_size), m_string(other.m_string) {
            other.m_size = 0;
            other.m_string = nullptr;
            instances_alive++;
        }

    // WITHOUT & it means that "this must return a brand new object". With it, it returns a reference.
    MyString& operator=(const MyString& other) {  // COPY ASSIGNMENT: a = b, a and b are both MyString. Existing object overwritten by LVALUE
        // RETURN BY REFERENCE because otherwise returns brand-new copy
        if (this != &other) {
            delete[] m_string;
            m_size = other.m_size;
            m_string = new char[m_size + 1];
            for (int i = 0; i < m_size; i++) {
                m_string[i] = other.m_string[i];
            }
            m_string[m_size] = '\0';
        }
        // one dies, one gets created so balance is 0

        return *this;
    }

    // some MyString var earlier...
    // var = function_that_returns_MyString()
    MyString& operator=(MyString&& other) noexcept { // MOVE ASSIGNMENT: a = function_that_returns_MyString(). Existing object overwritten by RVALUE
        if (this != &other) {
            delete[] m_string;
            m_size = other.m_size;
            m_string = other.m_string; // reassing pointer
            other.m_size = 0;
            other.m_string = nullptr;
        }

        return *this;
    }
    
    int get_length() const {return m_size;}

    const char* get_adress() const {return m_string;} // read-only adress

    const char* c_str() const {return m_string;}

    // read-only. Compiler knows which one to choose bc of const pairing - const objects get read-only, not const get modifiable.
    char operator[](const int idx) const {return m_string[idx];}

    char& operator[](const int idx) {return m_string[idx];}
    // returns reference to string[idx] - a specific value in string (automatically converts to memo adress)

    static int get_instances_alive() {return instances_alive;} // static funcs belong to the whole struct, not an individual object.
    
    void print_log() {
        logs_printed++;
        std::cout << "Size: " << m_size << std::endl;
        std::cout << "Data: " << c_str() << std::endl;
        std::cout << "Messages printed: " << logs_printed;
    }


};

// implement friended function outside of class brackets
std::ostream& operator<<(std::ostream& stream, const MyString& str) {
    stream << str.m_string;
    return stream; // return reference
}

int MyString::instances_alive = 0;
int MyString::logs_printed = 0;


int main() {
    return 0;
}