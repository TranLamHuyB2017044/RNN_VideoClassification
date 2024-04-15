import { useState } from "react"
import axios from "axios"
const URL_SERVER = "http://127.0.0.1:5000";
import Loading from './LoadingComponent/Loading'
function App() {
  const [previewVideo, setPreviewVideo] = useState()
  const [file, setFile] = useState()
  const [predictedClass, setPredictedClass] = useState(null)
  const [loading, setLoading] = useState(false)
  const handleClearVideo = () => {
    setPreviewVideo(null)
    setPredictedClass(null)
  }


  const handlePreviewVideo = (e) => {
    const files = e.target.files;
    setFile(files[0])
    const newUrl = files.length > 0 ? URL.createObjectURL(files[0]) : null;
    setPreviewVideo(newUrl);
  }


  const onSubmit = async (e) => {
    e.preventDefault();
    setPredictedClass(null)
    try {
      setLoading(true)
      console.log(file)
      const form = new FormData();
      form.append('video', file)
      const response = await axios.post(`/predict`, form, {
        baseURL: URL_SERVER,
        headers: {
          Accept: "application/json",
          "Content-Type": "multipart/form-data",
        }
      })
      if (response.status === 200) {
        setLoading(false)
        console.log(response.data.data)
        setPredictedClass(response.data.data)
      }
    } catch (error) {
      console.log(error)
      setLoading(false)
    }

  }



  return (
    <div className="flex flex-col items-center">
      <h1 className="text-center text-4xl text-[dodgerblue] mt-20">Video Classification  with CNN + RNN</h1>
      <p className="text-center  text-[dodgerblue] mt-5">Click to the image below to choose video</p>

      {previewVideo &&
        <div className="mt-5 flex flex-col items-center">
          <video className={`w-[320px] h-[200px]`} controls>
            <source src={previewVideo} type="video/mp4" />
          </video>
          {loading === false && <button onClick={handleClearVideo} className=" mt-5 px-8 py-2 border-[1px] bg-red-500 text-white rounded-md ">Clear video</button>}
        </div>}
      {loading ? <Loading /> :
        <form action="#" onSubmit={onSubmit}>
          <div className="mt-5">
            <label htmlFor="video" className="cursor-pointer">
              <img className='ml-10' width={100} height={100} src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcshTWid4p5RtNWu8Sy3ioMAqCRbDXwLsscB1770hXKA&s" alt="video-icon" />
            </label>
            <input type="file" id="video" className="hidden " onChange={handlePreviewVideo} />
          </div>
          {previewVideo && <button type="submit" className="px-16 py-3 rounded-md text-white border-[1px] bg-[dodgerblue] hover:bg-indigo-400 duration-300 to mt-5">Predict</button>}
        </form>}
        {predictedClass !==null && <div className="mt-4 flex">Video belong to class: <p className="text-red-500 ml-1">{predictedClass}</p></div>}
    </div>
  )
}

export default App
