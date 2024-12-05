# AndroidLogPipeline
A PySpark model used to classify Android Logs and detect anomalies in them
The "Logcat" tool in Android Studio is commonly used to view Android event logs, which are a record of system events and actions that take place on an Android device. In essence, they are a detailed log of everything that occurs within the operating system and applications, including user interactions, system processes, network activity, and potential errors. These logs can be accessed and analyzed for debugging and troubleshooting purposes. 
# Types of Andoid Log Levels
1. Verbose (Log.v)
Logs detailed and highly granular information, typically used for debugging purposes. Includes everything, even the least significant logs. This type of log is very noisy and should be used sparingly in production environments. Used for debugging specific scenarios during development to capture the complete flow of an operation.
2. Debug (Log.d)
Logs information useful for debugging during development. Less verbose than Verbose logs but provides sufficient context about the app's state and operations. Used for tracking application flow, values of variables, and non-critical operations.
3. Info (Log.i)
Logs general information about application events. Provides messages that indicate the normal functioning of the application. These logs are more concise and less detailed than Debug logs. Used for reporting key lifecycle events or application status (e.g., "User logged in").
4. Error (Log.e)
Logs error messages when something goes wrong in the application. Used for logging issues that need immediate attention, such as exceptions or failures in operations. Used for tracking critical failures, such as failed network requests, crashes, or unhandled exceptions.
5. Warning (Log.w)
Captures potential problems that are not yet errors but could lead to issues. Used for identifying areas that might need improvement or fixing in future releases.
6. Fatal/Assert (Log.wtf)
Logs critical problems that should "never happen" and often represent serious bugs. It is typically used for situations where recovery is impossible or undesirable. Corrupted data or application state inconsistency. Used for debugging extreme cases that indicate a breach of fundamental assumptions in your code.
