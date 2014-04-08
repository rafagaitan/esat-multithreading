/*******************************************************************************
   The MIT License (MIT)

   Copyright (c) 2014 Rafael Gaitan <rafa.gaitan@mirage-tech.com>
                                    http://www.mirage-tech.com

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.

   -----------------------------------------------------------------------------
   Additional Notes:

   Code for the Multithreading and Parallel Computing Course at ESAT
               -------------------------------
               |     http://www.esat.es      |
               -------------------------------

   more information of the course at:
       -----------------------------------------------------------------
       |  http://www.esat.es/estudios/programacion-multihilo/?pnt=621  |
       -----------------------------------------------------------------
**********************************************************************************/

#include <vector>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <algorithm>
#include <sstream>
#include <thread>
#include <memory>
#include <cstring>

// threading
namespace test_threading
{
    void hello()
    {
        std::cout << "Hello Concurrent" << std::endl;
    }

    void test_hello_concurrent()
    {
        std::thread t(hello);
        t.join();
    }
}
// auto
namespace test_auto
{
    class foo
    {
    public:
        foo():x(5) { }
        int x;

        foo& operator+=(const foo& rhs) { x += rhs.x; return *this; } 
        foo operator+(const foo& rhs) { foo ret; ret.x = x+rhs.x; return ret; } 
    };

    void test_auto()
    {
        auto i = 42;        // i is an int
        auto l = 42LL;      // l is an long long
        auto p = new foo(); // p is a foo*
        p->x = 5;
        std::cout << "i:" << i << ", l:" << l << ", p->x:" << p->x << std::endl;
        delete p;
    }
    void test_auto_collection()
    {
        std::map<std::string, std::vector<char>> map;
        map["hola"].push_back( 'h');
        map["hola"].push_back( 'o');
        map["hola"].push_back( 'l');
        map["hola"].push_back( 'a');
        map["mundo"].push_back('m');
        map["mundo"].push_back('u');
        map["mundo"].push_back('n');
        map["mundo"].push_back('d');
        map["mundo"].push_back('o');
        for(auto it = map.begin(); it != map.end(); ++it)
        {
            std::cout << it->first << ":";
            for(auto c = it->second.begin(); c != it->second.end(); ++c)
            {
                std::cout << *c;
            }
            std::cout << std::endl;
        }
    }
    auto test_auto_fn_bool() -> bool
    {
        return true;
    }
    auto test_auto_fn_int() -> int
    {
        return 42;
    }
    auto test_auto_fn_vector() -> std::vector<std::string>
    {
        std::vector<std::string> data;
        data.push_back("hola");
        data.push_back("mundo");
        return data;
    }
    template<typename T1, typename T2>
    auto test_auto_decltype(T1 t1, T2 t2) -> decltype(t1 + t2)
    {
        return (t1 + t2);
    }
    }

    //nullptr
namespace test_nullptr
{
    void foo(int* ) {}

    void test_nullptr()
    {
        int* p1 = NULL;
        int* p2 = 0;
        int* p3 = nullptr;
        if(p1 == p2 && p2 == p3)
        {
        }
        foo(nullptr);
        bool f = nullptr;
        if(f) { }
        //int k = NULL; if(k) { }// error: converting to non-pointer type 'int' from NULL [-Werror=conversion-null]
        //int i = nullptr; // error: A native nullptr can only be converted to bool or, using reinterpret_cast, to an integral type
    }
}

    //Range-based for loops
namespace test_range_based_loops
{
    void test_range_loop_collection()
    {
        std::map<std::string, std::vector<char>> map;
        map["hola"].push_back( 'h');
        map["hola"].push_back( 'o');
        map["hola"].push_back( 'l');
        map["hola"].push_back( 'a');
        map["mundo"].push_back('m');
        map["mundo"].push_back('u');
        map["mundo"].push_back('n');
        map["mundo"].push_back('d');
        map["mundo"].push_back('o');
        for(auto it: map)
        {
            std::cout << it.first << ":";
            for(auto c: it.second)
            {
                std::cout << c;
            }
            std::cout << std::endl;
        }
    }

    void test_range_loop_array()
    {
        int arr[] = {1, 2, 3, 4, 5};
        std::cout << "int arr[]:";
        for(auto & e : arr)
        {
            std::cout << e << " ";
            e = e * e;
        }
        std::cout << std::endl;
        std::cout << "int arr[](v*v):";
        for(auto & e : arr)
        {
            std::cout << e << " ";
        }
        std::cout << std::endl;
    }
}
//Override, final, delete
namespace test_override_final_delete
{
    #ifdef __clang__
    #pragma GCC diagnostic ignored "-Woverloaded-virtual"
    #endif
    class A
    {
    public:
        virtual ~A() {}
        virtual void f(short)
        {
            std::cout << "A::f" << std::endl;
        }
    };

    class B : public A
    {
    public:
        virtual ~B() {}
        virtual void f(int)
        {
            std::cout << "B::f" << std::endl;
        }
    };

    void test_hidden_virtual_method()
    {
        int i = 5;
        short s = 2;

        A *vA = new B();
        vA->f(i);
        vA->f(s);

        delete vA;
    }

    class C
    {
    public:
        virtual ~C() {}
        virtual void f(int) const
        {
            std::cout << "C::f " << std::endl;
        }
    };

    class D : public C
    {
    public:
        virtual ~D() {}
        virtual void f(int)
        {
            std::cout << "D::f" << std::endl;
        }
    };

    void test_unqualified_virtual_method()
    {
        int i = 5;
        short s = 2;

        C *vC = new D();
        vC->f(i);
        vC->f(s);

        delete vC;
    }

    class E
    {
    public:
        virtual ~E() {}
        virtual void f(int)
        {
            std::cout << "E::f" << std::endl;
        }
    };

    class F : public E
    {
    public:
        virtual ~F() {}
        virtual void f(int) override final {std::cout << "F::f" << std::endl;}
    };

    class G : public F
    {
    public:
        //virtual void f(int) override {std::cout << "G::f" << std::endl;}
    };

    void test_override_final_method()
    {
        int i = 5;
        short s = 2;

        E *vF = new F();
        vF->f(i);
        vF->f(s);

        delete vF;
    }

    class H
    {
    public:
        H(int i): i(i)
        {
            std::cout << "H(" << i << ")" << std::endl;
        }
#if defined(__clang__) || defined(__GNUG__)
        // non copyable
        H(const H&) = delete;
        // non assignable
        H& operator=(const H&) = delete;
#else
    private:
        H(const H&);
        H& operator=(const H&);
#endif
    public:
        // move constructible
        explicit H(H && rhs): i(std::move(rhs.i))
        {
            std::cout << "H(&&" << i << ")" << std::endl;
        }
        // move assignable
        H& operator=(H && rhs)
        {
            i = std::move(rhs.i);
            std::cout << "operator=(&&" << i << ")" << std::endl;
            return *this;
        }

        int i;
    };

    void test_delete_constructor_operators()
    {
        H h1(1), h2(2);
        //H i(h1);  // deleted
        //H j = h1; // deleted
        H k(std::move(h1)); // calls move constructor
        H l(3);
        l = std::move(h2);  // calls move assign
    }
    #ifdef __clang__
    #pragma GCC diagnostic warning "-Woverloaded-virtual"
    #endif
}
            
namespace test_strtok
{
    void test_strtok()
    {
        char buf1[512];
        char buf2[512];
        char *p, *q;
        strcpy(buf1, "esto es un test");
        strcpy(buf2, "esto no funciona seguro");
        p = strtok(buf1, " ");
        q = strtok(buf2, " ");
        while(p && q)
        {
            std::cout << p << ", " << q << std::endl;
            p = strtok(NULL, " ");
            q = strtok(NULL, " ");
        }
    }

    class Tokenizer
    {
    public:
        Tokenizer(const std::string& str):_str(str), _sstr(str) { }
        bool hasMoreTokens() { return !_sstr.eof(); }
        std::string nextToken() { std::string str; _sstr >> str; return str; }
        std::string _str;
        std::stringstream _sstr;
    };

    void test_tokenizer()
    {
        Tokenizer t1("esto es un test");
        Tokenizer t2("esto funciona con seguridad");
        while(t1.hasMoreTokens() && t2.hasMoreTokens())
        {
            std::cout << t1.nextToken() << ", " << t2.nextToken() << std::endl;
        }
    }
}
//Strongly-typed enums
namespace test_strong_typed_enums
{
	enum Alert { green, yellow, orange, red }; // traditional enum

	enum class Color { red, blue };   // scoped and strongly typed enum
	                                  // no export of enumerator names into enclosing scope
	                                  // no implicit conversion to int
	enum class TrafficLight { red, yellow, green };
    
    void test_strong_typed_enums()
    {
	    //Alert a = 7;              // error (as ever in C++)
	    //Color c = 7;              // error: no int->Color conversion

	    int a2 = red;             // ok: Alert->int conversion
	    int a3 = Alert::red;      // error in C++98; ok in C++11
	    //int a4 = blue;            // error: blue not in scope
	    //int a5 = Color::blue;     // error: not Color->int conversion

	    Color a6 = Color::blue;   // ok
        
        std::cout << "a2 (classic enum):" << a2 << std::endl;
        std::cout << "a3 (new enum):" << static_cast<int>(a3) << std::endl;
        std::cout << "a6 (new enum):" << static_cast<int>(a6) << std::endl;
    }

    enum class ColorChar : char { red, blue};

    void test_strong_typed_enums_type()
    {
        Color a6 = Color::blue;   // ok
        std::cout << "a6 (new enum):" << static_cast<char>(a6) << std::endl;
    }
}
//Smart pointers
namespace test_smart_ptr
{
    void foo(const std::string& id, int* p)
    {
        std::cout << id << ":" << *p << std::endl;
    }
    namespace test_unique_ptr
    {
        void test_unique_ptr()
        {
            std::unique_ptr<int> p1(new int(42));
            std::unique_ptr<int> p2 = std::move(p1); // transfer ownership
            
            if(p1)
                foo("p1",p1.get());
            
            (*p2)++;
            
            if(p2)
                foo("p2",p2.get());
        }
    }
    namespace test_shared_ptr
    {
        void bar(std::shared_ptr<int> p)
        {
            ++(*p);
        }
        void test_shared_ptr()
        {
            std::shared_ptr<int> p1(new int(42));
            std::cout << "reference count (p1):" << p1.use_count() << std::endl;
            std::shared_ptr<int> p2 = p1;
            std::cout << "reference count (p2 = p1):" << p1.use_count() << std::endl;
        
            bar(p1);
            foo("p2", p2.get());
        }
        class Object
        {
        public:
            Object(const std::string& str): _str(str)
            {
                std::cout << "Constructor " << _str << std::endl;
            }
            
            Object():_str("Object")
            {
                std::cout << "Default constructor " << _str << std::endl;
            }
            
            ~Object()
            {
                std::cout << "Destructor " << _str << std::endl;
            }
            
            Object(const Object& rhs):_str(rhs._str)
            {
                std::cout << "Copy constructor" << _str << std::endl;
            }
            std::string _str;
        private:
            Object(Object&&);
            Object& operator=(Object&&);

        };
        
        void test_make_shared_ptr()
        {
            {
                std::cout << "Create smart_ptr using make_shared..." << std::endl;
                std::shared_ptr<Object> p1 = std::make_shared<Object>("foo1");
                if(p1)
                    std::cout << "Create smart_ptr using make_shared: done." << std::endl;
                
            }
            {
                std::cout << "Create smart_ptr using new..." << std::endl;
                std::shared_ptr<Object> p2(new Object("foo2"));
                if(p2)
                    std::cout << "Create smart_ptr using new: done." << std::endl;
            }
        }
        
        void init(std::shared_ptr<Object> p, const std::string str)
        {
            p->_str = str;
        }
        
        std::string crash()
        {
            std::string lets_crash;
            lets_crash.at(0);
            return "crash";
        }
        
        void test_possible_leak()
        {
            {
                try {
                    std::cout << "Create smart_ptr using make_shared and crashing..." << std::endl;
                    init(std::make_shared<Object>("foo1"), crash());
                }
                catch(...)
                {
                    std::cout << "Crash using make_shared (leaks?)" << std::endl;
                }
            }
            {
                try
                {
                    std::cout << "Create smart_ptr using new and crashing..." << std::endl;
                    init(std::shared_ptr<Object>(new Object("foo2")), crash());
                }
                catch(...)
                {
                    std::cout << "Crash using new (leaks?)" << std::endl;
                }
            }
        }
    }
    namespace test_weak_ptr
    {
        void test_weak_ptr()
        {
            auto p = std::make_shared<int>(42);
            std::weak_ptr<int> wp = p;       
            {
                auto sp = wp.lock();
                if(sp)
                    foo("lock on weak pointr:", sp.get());
                else
                    std::cout << "is expired" << std::endl;
            }

            p.reset();       
            if(wp.expired())
                std::cout << "expired" << std::endl;

            {
                auto sp = wp.lock();
                if(sp)
                    foo("lock on weak pointr:", sp.get());
                else
                    std::cout << "is expired" << std::endl;
            }
        
        }
    }
}
//Lambdas
namespace test_lambdas
{
    std::vector<int> fill_data()
    {
#if defined(__clang__)
        std::vector<int> v{2, 3, 4, 5, 6, 7};
#else   
        int d[] = {2, 3, 4, 5, 6, 7};
        std::vector<int> v;
        v.assign(d, d+6);
#endif
        return v;
    }

    void test_lambda1()
    {
        auto v(fill_data());
        std::for_each(v.begin(), v.end(), [](int n) {std::cout << n << std::endl;});
    
        auto is_odd = [](int n) {return n%2==1;};
        auto pos = std::find_if(v.begin(), v.end(), is_odd);
        if(pos != std::end(v))
            std::cout << "first odd:" << *pos << std::endl;
    }
    
    void test_lambda_capture()
    {
        auto v(fill_data());
        int find_value = 5;
        std::cout << "find_value ptr=" << &find_value << std::endl;
        {
            auto pos = std::find_if(std::begin(v), std::end(v),[&](int n)
            {
                if(n == find_value)
                {
                    std::cout << n << " find_value ptr=" << &find_value << std::endl;
                    return true;
                }
                return false;
             });
            if(pos != std::end(v))
                std::cout << "found:" << *pos << std::endl;
        }
        {
            auto pos = std::find_if(std::begin(v), std::end(v), [=](int n)
            {
                if(n == find_value)
                {
                    std::cout << n << " find_value ptr=" << &find_value << std::endl;
                    return true;
                }
                return false;
            });
            if(pos != std::end(v))
                std::cout << "found:" << *pos << std::endl;
        }
    }
    
    void test_recursive_lambda()
    {
#if !defined(__clang__)
        std::function<int(int)> lfib =
           [&lfib](int n) {return n < 2 ? 1 : lfib(n-1) + lfib(n-2);};
        //auto lfib =
        //    [&lfib](int n) {return n < 2 ? 1 : lfib(n-1) + lfib(n-2);};
        std::cout << "fibonacci (5)=" << lfib(5) << std::endl;
#endif
    }
}
//non-member begin() and end()
//static_assert and type traits
//Move semantics
namespace test_move_semantics
{
class Object
{
public:
    explicit Object(const std::string& str): _str(str)
    {
        std::cout << "Constructor " << _str << std::endl;
    }  
    Object():_str("Object")
    {
        std::cout << "Default constructor " << _str << std::endl;
    }   
    ~Object()
    {
        std::cout << "Destructor " << _str << std::endl;
    }
    explicit Object(const Object& rhs):_str(rhs._str)
    {
        std::cout << "Copy constructor " << _str << std::endl;
    }
    explicit Object(Object&& rhs): _str(std::move(rhs._str)) 
    { 
        std::cout << "Move Copy Constructor " << _str << std::endl;
    }
    Object& operator=(const Object& rhs)
    {
        _str = rhs._str;
        std::cout << "Assing operator " << _str << std::endl;
        return *this;
    }
    Object& operator=(Object&& rhs)
    {
        _str = std::move(rhs._str);
        std::cout << "Move assing operator " << _str << std::endl;
        return *this;
    }
    std::string _str;
};

    void test_move_semantics()
    {
        Object o1("o1");
        Object o2(o1);
        Object o3(Object("o3"));
        Object o3_(std::move(o3));
        Object o4("o4");
        Object o4_("o4_");
        o4 = o4_;
        Object o5("o5");
        o5 = std::move(o4);
    }
}



int main(int , char**)
{
    // test_threading
    std::cout << "Test Threading:" << std::endl;
    test_threading::test_hello_concurrent();
    std::cout << "-----------------------------" << std::endl << std::endl;


    // test_strtok
    std::cout << "Test Tokenizer:" << std::endl;
    test_strtok::test_strtok();
    test_strtok::test_tokenizer();
    std::cout << "-----------------------------" << std::endl << std::endl;

    // auto
    std::cout << "Test Auto:" << std::endl;
    test_auto::test_auto();
    test_auto::test_auto_collection();
    auto bValue = test_auto::test_auto_fn_bool();
    auto iValue = test_auto::test_auto_fn_int();
    auto vValue = test_auto::test_auto_fn_vector();
    test_auto::foo f1;
    test_auto::foo f2; 
    auto sumValue = test_auto::test_auto_decltype(f1, f2);

    std::cout << "bValue: " << bValue << std::endl;
    std::cout << "iValue: " << iValue << std::endl;
    std::cout << "vValue typeof:" << typeid(vValue).name() << std::endl;
    std::cout << "vValue:[";
    std::copy(vValue.begin(), vValue.end(), std::ostream_iterator<std::string>(std::cout, ", "));
    std::cout << "]" << std::endl;
    std::cout << "sumValue: " << sumValue.x << " typeof:" << typeid(sumValue).name() << std::endl;
    std::cout << "-----------------------------" << std::endl << std::endl;

    // nullptr
    std::cout << "Test nullptr:" << std::endl;
    test_nullptr::test_nullptr();
    std::cout << "-----------------------------" << std::endl << std::endl;

    // range based loops
    std::cout << "Test range based loops:" << std::endl;
    test_range_based_loops::test_range_loop_collection();
    test_range_based_loops::test_range_loop_array();
    std::cout << "-----------------------------" << std::endl << std::endl;

    std::cout << "Test override, final, =delete:" << std::endl;
    test_override_final_delete::test_hidden_virtual_method();
    test_override_final_delete::test_unqualified_virtual_method();
    test_override_final_delete::test_override_final_method();
    test_override_final_delete::test_delete_constructor_operators();
    std::cout << "-----------------------------" << std::endl << std::endl;


    std::cout << "Test strongly-typed enums" << std::endl;
    test_strong_typed_enums::test_strong_typed_enums();
    test_strong_typed_enums::test_strong_typed_enums_type();
    std::cout << "-----------------------------" << std::endl << std::endl;
    
    std::cout << "Test smart pointers" << std::endl;
    test_smart_ptr::test_unique_ptr::test_unique_ptr();
    test_smart_ptr::test_shared_ptr::test_shared_ptr();
    test_smart_ptr::test_shared_ptr::test_make_shared_ptr();
    test_smart_ptr::test_shared_ptr::test_possible_leak();
    test_smart_ptr::test_weak_ptr::test_weak_ptr();
    std::cout << "-----------------------------" << std::endl << std::endl;
    
    std::cout << "Test lambda" << std::endl;
    test_lambdas::test_lambda1();
    test_lambdas::test_lambda_capture();
    test_lambdas::test_recursive_lambda();
    std::cout << "-----------------------------" << std::endl << std::endl;
    
    std::cout << "Test Move semantics" << std::endl;
    test_move_semantics::test_move_semantics();
    std::cout << "-----------------------------" << std::endl << std::endl;
    
    return EXIT_SUCCESS;
}

