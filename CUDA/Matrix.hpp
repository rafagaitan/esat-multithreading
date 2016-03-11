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
#pragma once

#include <ctime>
#include <algorithm>
#include <vector>
#include <iomanip>


template<class T>
struct HostMatrix
{
    HostMatrix(int width,int height)
        :width_(width)
        ,height_(height)
        ,size_(width_*height_)
        ,elements_(size_)
    {
        std::fill(elements_.begin(),elements_.end(),0.0f);
    }
    ~HostMatrix()
    {
    }

    HostMatrix(const HostMatrix& rhs)
        :width_(rhs.width_)
        ,height_(rhs.height_)
        ,size_(rhs.size_)
        ,elements_(rhs.elements_)
    {
    }
    HostMatrix& operator=(const HostMatrix& rhs)
    {
        width_    = rhs.width_;
        height_   = rhs.height_;
        size_     = rhs.size_;
        elements_ = rhs.elements_;
        return *this;
    }

    void fillWithRandomData()
    {
        srand (static_cast <unsigned> (time(0)));
        for(int j=0;j<height_;++j)
        {
            for(int i=0;i<width_;++i)
            {
                elements_[j*width_+i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/256.0));
            }
        }
    }

    size_t sizeInBytes()
    {
        return size_*sizeof(T);
    }

    void identity()
    {
        int i=0;
        for(int j=0;j<height_;++j)
        {
            if(i>=width_)
                break;
            elements_[j*width_+i] = 1.0;
            i++;
        }
    }

    void fill_diagonal(T value)
    {
        int i=0;
        for(int j=0;j<height_;++j)
        {
            if(i>=width_)
                break;
            elements_[j*width_+i] = value;
            i++;
        }
    }

    HostMatrix<T> operator*(const HostMatrix<T>& m)
    {
        HostMatrix<T> mult(m.width_,height_);
        for(int row=0; row<mult.height_; ++row)
        for(int col=0; col<mult.width_; ++col)
        {
            T v = 0.0;
            for(int k=0; k<width_; ++k)
            {
                v+=getData(row,k)*m.getData(k,col);
            }
            mult.setData(row,col,v);
        }
        return mult;
    }

    T getData(int row, int col) const
    {
        return elements_[row * width_ + col];
    }

    void setData(int row, int col, T data)
    {
        elements_[row * width_ + col] = data;
    }

    void print(std::ostream& os)
    {
        for(int j=0;j<height_;++j)
        {
            os <<"| ";
            for(int i=0;i<width_;++i)
            {
                os << std::setw(10) << elements_[j*width_+i];
            }
            os <<" |" << std::endl;
        }
    }

    operator const T* ()
    {
        return elements_.data();
    }

    operator const void* ()
    {
        return reinterpret_cast<const void*>(elements_.data());
    }

    operator T* ()
    {
        return elements_.data();
    }


    int            width_;
    int            height_;
    int            size_;
    std::vector<T> elements_;
};


