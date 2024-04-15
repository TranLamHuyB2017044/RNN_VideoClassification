export default function Loading() {
    return (
        <div className="flex items-center justify-center mt-12">
        <div className="flex space-x-2">
          <div className="h-5 w-5 bg-red-500 rounded-full animate-bounce"></div>
          <div className="h-5 w-5 bg-green-500 rounded-full animate-bounce"></div>
          <div className="h-5 w-5 bg-blue-500 rounded-full animate-bounce"></div>
        </div>
      </div>
    )
  }