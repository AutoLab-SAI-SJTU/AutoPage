const config = {
  "videos": {
    "Train Tasks": ["public/videos/box_high_level.mp4", "public/videos/box_low_level.mp4", "public/videos/couch_moving_high_level.mp4", "public/videos/couch_moving_low_level.mp4", "public/videos/stack_high_level.mp4", "public/videos/stack_low_level.mp4", "public/videos/drawer_high_level.mp4", "public/videos/drawer_low_level.mp4"],
    "Test Tasks": ["public/videos/box_high_level_test.mp4", "public/videos/box_low_level_test.mp4", "public/videos/couch_moving_high_level_test.mp4", "public/videos/couch_moving_low_level_test.mp4", "public/videos/stack6_high_level.mp4", "public/videos/stack6_low_level.mp4", "public/videos/drawer_high_level_test.mp4", "public/videos/drawer_low_level_test.mp4"],
  }
}
let visualize_hard_init_states = false;
let visualize_easy_init_states = false;
$(document).ready(function () {
  // var f = window.onscroll;
  let sectionBars = $(".barWrapper");
  $(window).scroll(function () {
    let py = window.pageYOffset;
    let buffer = window.innerHeight / 2;
    $(".barWrapper").removeClass("inview");
    for (let i = sectionBars.length - 1; i >= 0; i--) {
      //55 is anchor css top magnitude
      const elem = document.getElementById(
        sectionBars[i].attributes.href.value.slice(1)
      );
      if (py > elem.offsetTop - buffer + 55) {
        $(sectionBars[i]).addClass("inview");
        break;
      }
    }
  });
  $(".sidebarsWrapper").hover(
    function () {
      $(".sidebars span").css("opacity", 1);
    },
    function () {
      $(".sidebars span").css("opacity", 0);
    }
  );
  window.scroll();

  // const srcs = document.getElementsByClassName("abstractexecutable")[0].getElementsByTagName("source")
  // const vids = document.getElementsByClassName("abstractexecutable")[0].getElementsByTagName("video")
  // document.querySelector("source").src = "http://example.com/new_url.mp4"

  // function setVideos(cond) {
  //   const videos = config.videos[cond]
  //   if (cond === "Train Tasks") {

  //   }
  //   for (let i = 0; i < vids.length; i+=1) {
  //     const vidElem = vids[i];
  //     const source = vidElem.getElementsByTagName("source")[0]
  //     vidElem.pause();
  //     source.setAttribute("src", videos[i]);
  //     vidElem.load();
  //   }
  // }
  // setVideos("Test Tasks")

  $("#visualize-hard-init-states").on('click', function (e) {
    visualize_hard_init_states = !visualize_hard_init_states;
    const el = document.getElementById("visualize-hard-init-states");
    if (!visualize_hard_init_states) {
      el.textContent = "Visualize Initial States"
      let videos = $("#hard-videos video");
    for (let i = 0; i < videos.length; i++) {
      const element = videos[i];
      element.play();
    };
    } else {
      el.textContent = "Show Policy Trajectories"
      let videos = $("#hard-videos video");
    for (let i = 0; i < videos.length; i++) {
      const element = videos[i];
      element.pause();
      element.currentTime = 0;
    };
    }
  })
  $("#visualize-easy-init-states").on('click', function (e) {
    visualize_easy_init_states = !visualize_easy_init_states;
    const el = document.getElementById("visualize-easy-init-states");
    if (!visualize_easy_init_states) {
      el.textContent = "Visualize Initial States"
      let videos = $("#easy-videos video");
    for (let i = 0; i < videos.length; i++) {
      const element = videos[i];
      element.play();
    };
    } else {
      el.textContent = "Show Policy Trajectories"
      let videos = $("#easy-videos video");
    for (let i = 0; i < videos.length; i++) {
      const element = videos[i];
      element.pause();
      element.currentTime = 0;
    };
    }
  })
});
