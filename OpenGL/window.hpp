/*
Author: Ismael Castellanos Ruiz
Data: 11/01/2014
Contact: iscaru1988@gmail.com
*/

#ifndef __WINDOW_H__
#define __WINDOW_H__ 1

#include "Export.hpp"
#include "types.hpp"
#include <glm/glm.hpp>

struct GLFWwindow;

//!A class to handle the Window.
/*! It can be used to open a context for hardware accelerated applications and other few things...*/
class HDK_EXPORT Window{

public:

    Window();

    //! An enum for the screen types.
    enum ScreenTypeGLFW {
        FULLSCREEN = 1 << 0, /*!< Enum value to set the fullscreen mode. */
        VSYNC = 1 << 1, /*!< Enum value to set the windowed mode. */
    };

    ~Window();

    //!Inits the window with the specified width, height and mode (See the ScreenType enum).
    /*!
    \param width Specifies the window width.
    \param height Specifies the window height.
    \param mode Specifies if it will be windowed or fullscreen, by default it is windowed (See the ScreenType enum).
    */
    bool Init(int32 width, int32 height, const char* title, ubyte screenOpcs = 0);

    //!Processes all the events since the last call of this function.. IT IS CALLED AUTOMATICALLY WHEN WE CALL THE swap() FUNCTION.
    //void ProcessEvents();

    //!Swaps the buffers
    /*!Swaps the back buffer for the front buffer, it will also poll events.*/
    void Swap();

    //!Returns if the close close button of the window has been clicked
    int32 ShouldClose();

    //!Sets the clear color.*/
    /*!Sets the new RGBA color that will be used to clear the screen.*/
    void SetClearColor(glm::vec4& newClearColor);

    //!This functions enables the vertical synchronization.
    /*!This function synchronizes every swap of the buffers with the screen refresh. It limits the game fps to the screen refresh rate*/
    void EnableVsync();

    //!This functions disables the vertical synchronization.
    /*!This function does't set a fps limit but can cause tearing*/
    void DisableVsync();

    //!Ends the application and closes the context.
    /*!It will free the momory reserved for the window context and close the program.*/
    void Finish();

    //!Sets a callback function for key events.
    /*It will call the parametrized funcion every time a key is pressed, the function must be "void funcName(int, int, int)".*/
    void setWindowKeyCallback(CBFunctionKey* cbFunction);

private:


    //friend class Engine;
    GLFWwindow*       windowHandler;
    CBFunctionKey*    keyCallback; 

    static void setGLFWKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

};

#endif