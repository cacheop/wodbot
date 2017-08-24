angular.module('demo', [])
.controller('data', function($scope, $http) {

	var dataObj = {"gen_length":150};	

	var res = $http.post('http://54.200.82.138:5000/model', dataObj);
	
	res.success(function(data, status, headers, config) {
		$scope.message = data;
	});
	
	res.error(function(data, status, headers, config) {
		alert( "failure message: " + JSON.stringify({data: data}));
        });
});


