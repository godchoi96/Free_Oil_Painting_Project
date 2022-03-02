function setThumbnail(input) {
    if (input.files && input.files[0]) {
        // 예외처리 부분을 생각해야 합니다. 예를 들면, 이미지 확장자가 아니라던가 하는 부분에 대해서 체크해야 합니다.
        // FileReader 인스턴스 생성
        const reader = new FileReader()
        // 이미지가 로드가 된 경우
        reader.onload = e => {
            const previewImage = document.getElementById("preview-image")
            previewImage.src = e.target.result
        }
        // reader가 이미지 읽도록 하기
        reader.readAsDataURL(input.files[0])
    }
}

const inputImage = document.getElementById("file")
inputImage.addEventListener("change", e => {
    readImage(e.target)
})