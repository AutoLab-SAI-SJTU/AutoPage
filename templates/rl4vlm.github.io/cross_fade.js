function cross_fade_image(image_node, new_link, group_node, transition_duration) {
    var original_x = +image_node.attr('x');
    var original_y = +image_node.attr('y');
    var original_width = +image_node.attr('width');
    var original_height = +image_node.attr('height');
    
    var new_image = group_node.append('svg:image')
        .attr('id', 'transition_' + image_node.id)
        .attr('width', original_width)
        .attr('height', original_height)
        .attr('xlink:href', image_node.attr('xlink:href'))
        .attr('x', original_x)
        .attr('y', original_y)
        .style('opacity', 1.0);
    
    image_node
        .attr('xlink:href', new_link);
        
    new_image
        .transition()
        .duration(transition_duration)
        .style('opacity', 0.0)
        .attr('x', original_x)
        .attr('y', original_y)
        .remove();
}