$(document).ready(function() {

	$('form').on('submit', function(event) {

		$.ajax({
			data : {
				inp : $('#inp').val()


			},
			type : 'POST',
			url : '/chatbot/process'
		})
		.done(function(data) {

			if (data.error) {

				$('#successAlertInp').hide();
				$('#successAlertAnswer').hide();
			}
			else {
			    $('#successAlertInp').text(data.inp).show();
				$('#successAlertAnswer').text(data.answer).show();

			}

		});

		event.preventDefault();

	});

});