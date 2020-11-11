
function getBotResponse() {
  var rawText = $("#inp").val();
  var userHtml = '<p class="userText"><span>' + rawText + "</span></p>";
  $("#textInput").val("");
  $("#chatbox").append(userHtml);
  document
    .getElementById("userInput")
    .scrollIntoView({ block: "start", behavior: "smooth" });
  $.get("/chatbot/process", { inp: rawText }).done(function(answer) {
    var botHtml = '<p class="botText"><span>' + answer + "</span></p>";
    $("#chatbox").append(botHtml);
    document
      .getElementById("userInput")
      .scrollIntoView({ block: "start", behavior: "smooth" });
  });
}
$("#textInput").keypress(function(e) {
  if (e.which == 13) {
    getBotResponse();
  }
});
