// Written by Dor Verbin, October 2021
// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(id) {
    if (id.endsWith('After')) {
        id = id.slice(0, -5); // 'After' has 5 characters
    }
    var leftVideo = document.getElementById(id);
    var rightVideo = document.getElementById(id + 'After');
    var videoMerge = document.getElementById(id + 'Merge');
    var videoContainer = document.getElementById(id + 'Div');
    var videoSpinner = document.getElementById(id + 'Spinner');
    var mergeContext = videoMerge.getContext('2d');

    if (
        leftVideo.readyState > HTMLMediaElement.HAVE_CURRENT_DATA &&
        rightVideo.readyState > HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
        videoSpinner.style.display = 'none';
        // videoSpinner.style.border = 'none';
        // videoSpinner.style.height = '0px';

        var position = 0.5;
        var vertical = 0.5;

        var vidWidth = videoMerge.width;
        var vidHeight = videoMerge.height;
        var ratio = leftVideo.videoWidth / vidWidth;

        // leftVideo.removeEventListener('play');
        // rightVideo.removeEventListener('play');

        leftVideo.pause();
        leftVideo.currentTime = 0;
        leftVideo.play();

        rightVideo.pause();
        rightVideo.currentTime = 0;
        rightVideo.play();

        function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = (e.clientX - bcr.x) / bcr.width;
            vertical = (e.clientY - bcr.y) / bcr.height;
        }
        function trackLocationTouch(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = (e.touches[0].clientX - bcr.x) / bcr.width;
            vertical = (e.touches[0].clientY - bcr.y) / bcr.height;
        }

        videoMerge.addEventListener('mousemove', trackLocation, false);
        videoMerge.addEventListener('touchstart', trackLocationTouch, false);
        videoMerge.addEventListener('touchmove', trackLocationTouch, false);

        function drawLoop() {
            mergeContext.drawImage(
                leftVideo,
                0,
                0,
                vidWidth * ratio,
                vidHeight * ratio,
                0,
                0,
                vidWidth,
                vidHeight
            );
            var colStart = (vidWidth * position).clamp(0.0, vidWidth);
            var colWidth = (vidWidth - vidWidth * position).clamp(
                0.0,
                vidWidth
            );
            mergeContext.drawImage(
                rightVideo,
                colStart * ratio,
                0,
                colWidth * ratio,
                vidHeight * ratio,
                colStart,
                0,
                colWidth,
                vidHeight
            );
            // videoMerge.setAttribute(
            //     'animation_id',
            //     requestAnimationFrame(drawLoop)
            // );
            requestAnimationFrame(drawLoop);
            // videoMerge.setAttribute('darwLoop', drawLoop);

            var arrowLength = 0.09 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = 0.075 * vidHeight;
            var arrowWidth = 0.007 * vidHeight;
            var currX = vidWidth * position;

            // Draw circle
            mergeContext.arc(
                currX,
                arrowPosY,
                arrowLength * 0.7,
                0,
                Math.PI * 2,
                false
            );
            mergeContext.fillStyle = '#FFD79340';
            mergeContext.fill();
            // mergeContext.strokeStyle = "#444444";
            // mergeContext.stroke()

            // Draw border
            mergeContext.beginPath();
            mergeContext.moveTo(vidWidth * position, 0);
            mergeContext.lineTo(vidWidth * position, vidHeight);
            mergeContext.closePath();
            mergeContext.strokeStyle = '#AAAAAA';
            mergeContext.lineWidth = 5;
            mergeContext.stroke();

            // 添加以下代码来绘制文本

            // 保存当前上下文状态
            mergeContext.save();

            // 设置字体样式
            var textHeight = 20;
            mergeContext.font = '800 ' + textHeight + "px 'Jost'"; // 更换字体进行测试
            mergeContext.fillStyle = '#000000'; // 使用黑色

            // 设定开始的高度
            var startY = arrowPosY;

            // 在左侧绘制 "4K4D" 文本
            var text4K4D = '4K4D';
            var text4K4DWidth = mergeContext.measureText(text4K4D).width;
            mergeContext.fillText(
                text4K4D,
                vidWidth * position - text4K4DWidth - 50,
                startY + (textHeight / 5) * 2
            );

            // 在右侧绘制 "ENeRF" 文本
            var textENeRF = id;
            mergeContext.fillText(
                textENeRF,
                vidWidth * position + 50,
                startY + (textHeight / 5) * 2
            );

            // 恢复上下文状态
            mergeContext.restore();

            // Draw arrow
            mergeContext.beginPath();
            mergeContext.moveTo(currX, arrowPosY - arrowWidth / 2);

            // Move right until meeting arrow head
            mergeContext.lineTo(
                currX + arrowLength / 2 - arrowheadLength / 2,
                arrowPosY - arrowWidth / 2
            );

            // Draw right arrow head
            mergeContext.lineTo(
                currX + arrowLength / 2 - arrowheadLength / 2,
                arrowPosY - arrowheadWidth / 2
            );
            mergeContext.lineTo(currX + arrowLength / 2, arrowPosY);
            mergeContext.lineTo(
                currX + arrowLength / 2 - arrowheadLength / 2,
                arrowPosY + arrowheadWidth / 2
            );
            mergeContext.lineTo(
                currX + arrowLength / 2 - arrowheadLength / 2,
                arrowPosY + arrowWidth / 2
            );

            // Go back to the left until meeting left arrow head
            mergeContext.lineTo(
                currX - arrowLength / 2 + arrowheadLength / 2,
                arrowPosY + arrowWidth / 2
            );

            // Draw left arrow head
            mergeContext.lineTo(
                currX - arrowLength / 2 + arrowheadLength / 2,
                arrowPosY + arrowheadWidth / 2
            );
            mergeContext.lineTo(currX - arrowLength / 2, arrowPosY);
            mergeContext.lineTo(
                currX - arrowLength / 2 + arrowheadLength / 2,
                arrowPosY - arrowheadWidth / 2
            );
            mergeContext.lineTo(
                currX - arrowLength / 2 + arrowheadLength / 2,
                arrowPosY
            );

            mergeContext.lineTo(
                currX - arrowLength / 2 + arrowheadLength / 2,
                arrowPosY - arrowWidth / 2
            );
            mergeContext.lineTo(currX, arrowPosY - arrowWidth / 2);

            mergeContext.closePath();

            mergeContext.fillStyle = '#AAAAAA';
            mergeContext.fill();
        }
        // videoMerge.setAttribute(
        //     'animation_id',
        //     requestAnimationFrame(drawLoop)
        // );
        requestAnimationFrame(drawLoop);
    }
}

Number.prototype.clamp = function (min, max) {
    return Math.min(Math.max(this, min), max);
};

function resizeAndPlay(video) {
    var id = video.id;
    if (id.endsWith('After')) {
        id = id.slice(0, -5); // 'After' has 5 characters
    }

    var leftVideo = document.getElementById(id);
    var rightVideo = document.getElementById(id + 'After');
    var videoMerge = document.getElementById(id + 'Merge');

    var vidWidth = $(video).parent().width();
    var vidHeight = (video.videoHeight / video.videoWidth) * vidWidth;

    if (vidWidth > 0) {
        videoMerge.width = vidWidth;
        videoMerge.height = vidHeight;
    }

    playVids(video.id);
}
