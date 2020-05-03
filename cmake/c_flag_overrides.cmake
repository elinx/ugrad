# gtest prefer using CRT as static library while cmake always defaults
# to use shared libraries, which will cause linking error:
# "LNK2038: mismatch detected for 'RuntimeLibrary': value 'MTd_StaticDebug' doesn't match value 'MDd_DynamicDebug' in xxx.obj"
# the following code derived try to fix this issue.
# ref1: https://gitlab.kitware.com/cmake/community/-/wikis/FAQ#how-can-i-build-my-msvc-application-with-a-static-runtime
# ref2: googletest-1.10.0/googletest/cmake/internal_utils.cmake
if(MSVC)
  set(CMAKE_C_FLAGS_DEBUG_INIT "/D_DEBUG /MTd /Zi /Ob0 /Od /RTC1")
  set(CMAKE_C_FLAGS_MINSIZEREL_INIT     "/MT /O1 /Ob1 /D NDEBUG")
  set(CMAKE_C_FLAGS_RELEASE_INIT        "/MT /O2 /Ob2 /D NDEBUG")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "/MT /Zi /O2 /Ob1 /D NDEBUG")
endif()
