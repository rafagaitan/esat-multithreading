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

#include <string>
#include <vector>
#include <sstream>
#include <iostream>

class ArgumentParser
{
public:
    ArgumentParser(int argc, char* argv[]):
        arguments()
    {
        for(int i=1;i<argc;i++)
            arguments.push_back(argv[i]);
    }
    size_t size()  { return arguments.size();  }
    bool   empty() { return arguments.empty(); }

    static bool isOption(const std::string& arg)
    {
        return (arg[0] == '-');
    }

    template<typename... Args>
    bool read(const std::string& opt, Args&... params)
    {
        if(!isOption(opt))
            return false;
        // find the initial option that matches
        auto argItr_init = std::find_if(arguments.begin(),arguments.end(),
            [&,this](const std::string& arg) -> bool { return (isOption(arg) && arg == opt); });
        if(argItr_init == arguments.end()) return false;
        // find the next option
        auto argItr_end = std::find_if(std::next(argItr_init),arguments.end(),&ArgumentParser::isOption);
        // create a temporal args to parse it afterwards
        ArgumentList parseArgs(argItr_init,argItr_end);
        // erase it from the original arguments
        arguments.erase(argItr_init,argItr_end);
        // parse parameters
        return _read(parseArgs.begin(),parseArgs.end(), params...);
    }
    typedef std::vector<std::string> ArgumentList;
    ArgumentList arguments;
private:


    template<typename T, typename... Args>
    bool _read(const ArgumentList::iterator& begin, const ArgumentList::iterator& end, T& value, Args&... params)
    {
        auto argItr = std::next(begin);
        if(argItr == end || isOption(*argItr))
            return false;
        std::stringstream sstr; sstr << *argItr;
        sstr >> value;
        return _read(argItr, end, params...);
    }
    template<typename T>
    bool _read(const ArgumentList::iterator& begin, const ArgumentList::iterator& end, T& value)
    {
        auto argItr = std::next(begin);
        if(argItr == end || isOption(*argItr))
            return false;
        std::stringstream sstr; sstr << *argItr;
        sstr >> value;
        return true;
    }
};
