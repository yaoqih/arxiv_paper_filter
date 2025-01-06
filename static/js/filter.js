const itemsPerPage = 6; // 每页显示6篇论文
let currentPage = 1;
let filterData = {};
let globalStartDate, globalEndDate, globalfilterLabels ,globalStartScore=-1,globalEndScore=99999;
let paperDataGlobal = [];
const INIT_KEYWORD_COUNT = 10;
const EXPAND_KEYWORD_COUNT = 10;

const formatDate = (date) => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
};
$.ajax({
    url: "/get_keywords", // 确保 URL 正确指向你的服务器
    type: "POST",
    contentType: 'application/json',
    data: JSON.stringify({dataPath:dataPath}),
    success: function (data) {
        // 成功获取到数据后，将数据赋值给 filterData 变量
        filterData = data;
        let allDates = [];
        Object.values(filterData).forEach(arr => {
            Object.values(arr).forEach(item => {
                if (item.dates) {
                    allDates = allDates.concat(item.dates);
                }
            });
        });

        // 去重并排序
        allDates = [...new Set(allDates)].sort();

        const minDate = new Date(allDates[0]);
        const maxDate = new Date(allDates[allDates.length - 1]);

        $("#slider-range").slider({
            range: true,
            min: minDate.getTime(),
            max: maxDate.getTime(),
            step: 86400000, // 一天的毫秒数
            values: [minDate.getTime(), maxDate.getTime()],
            slide: function (event, ui) {
                const startDate = new Date(ui.values[0]);
                const endDate = new Date(ui.values[1]);

                // 格式化日期为 YYYY-MM-DD

                globalStartDate = formatDate(startDate);
                globalEndDate = formatDate(endDate);
                $("#date-range").text(
                    formatDate(startDate) + " , " + formatDate(endDate)
                );
                filterByDateRange(filterData, globalStartDate, globalEndDate);
            }
        });
        globalStartDate = formatDate(minDate);
        globalEndDate = formatDate(maxDate);
        filterByDateRange(filterData, formatDate(minDate), formatDate(maxDate));

        // 设置初始显示的日期范围
        $("#date-range").text(
            allDates[0] + " , " + allDates[allDates.length - 1]
        );
    },
    error: function (jqXHR, textStatus, errorThrown) {
        // 处理错误情况
        console.error("获取数据失败:", textStatus, errorThrown);
    }
});


function renderPaperCards(pageData) {
    const container = document.getElementById('paperContainer');
    
    // 使用 DocumentFragment 优化性能
    const fragment = document.createDocumentFragment();
    
    pageData.forEach(paper => {
        const div = document.createElement('div');
        // 使用模板字符串简化拼接
        div.innerHTML = `
            <div class="card h-100 shadow-sm">
                <div class="card-body">
                    <h5 class="card-title text-primary mb-3">${escapeHtml(paper.title)}</h5>
                    <p class="card-text text-muted mb-2">
                        <i class="fas fa-users"></i> ${paper.authors!='[]' ? escapeHtml(paper.authors): ''}
                    </p>
                    <p class="card-text text-muted mb-2">
                        <i class="far fa-calendar-alt"></i> ${escapeHtml(paper.published)}
                    </p>
                    <p class="card-text abstract-text">${escapeHtml(paper.abstract)}</p>
                    <div class="mb-2">
                        ${paper.labels.map(label =>
                            `<span class="badge bg-secondary me-1">${escapeHtml(label)}</span>`
                        ).join('')}
                    </div>
                    <div class="d-flex justify-content-between align-items-center mt-3">
                        <div>
                            <span class="badge bg-success">评分: ${escapeHtml(String(paper.score))}</span>
                        </div>
                        <div class="d-flex gap-2">
                            ${paper.wechat_url ? 
                                `<a href="${escapeHtml(paper.wechat_url)}" class="btn btn-outline-primary btn-sm" target="_blank">论文简读</a>` 
                                : ''
                            }
                            <a href="http://111.231.28.98:2333/?q=${encodeURIComponent(paper.id)}" class="btn btn-outline-primary btn-sm" target="_blank">
                                Arxiv Connect
                            </a>
                            <a href="${escapeHtml(paper.id)}" class="btn btn-outline-primary btn-sm" target="_blank">
                                阅读论文
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        `;
        fragment.appendChild(div);
    });

    // 清空容器并一次性添加所有内容
    container.innerHTML = '';
    container.appendChild(fragment);
}

// 添加 HTML 转义函数以防止 XSS 攻击
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


// 生成分页控件
function renderPagination() {
    const totalPages = Math.ceil(paperDataGlobal.length / itemsPerPage);
    const pagination = document.getElementById('pagination');
    pagination.innerHTML = '';

    // 上一页按钮
    const prevBtn = `
        <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${currentPage - 1})">上一页</a>
        </li>
    `;

    // 页码按钮
    let pageButtons = '';
    const showPages = 2; // 当前页码前后显示的页数

    for (let i = 1; i <= totalPages; i++) {
        // 显示首页、末页、当前页附近的页码，其他用省略号替代
        if (
            i === 1 || // 首页
            i === totalPages || // 末页
            (i >= currentPage - showPages && i <= currentPage + showPages) // 当前页附近
        ) {
            pageButtons += `
                <li class="page-item ${currentPage === i ? 'active' : ''}">
                    <a class="page-link" href="#" onclick="changePage(${i})">${i}</a>
                </li>
            `;
        } else if (
            i === currentPage - showPages - 1 || 
            i === currentPage + showPages + 1
        ) {
            // 添加省略号
            pageButtons += `
                <li class="page-item disabled">
                    <span class="page-link">...</span>
                </li>
            `;
        }
    }

    // 下一页按钮
    const nextBtn = `
        <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${currentPage + 1})">下一页</a>
        </li>
    `;

    pagination.innerHTML = prevBtn + pageButtons + nextBtn;
}


// 切换页面
function changePage(page) {
    if (page < 1 || page > Math.ceil(paperDataGlobal.length / itemsPerPage)) return;

    currentPage = page;
    const start = (page - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const pageData = paperDataGlobal.slice(start, end);

    renderPaperCards(pageData);
    renderPagination();
}
// 渲染层级关键词
function filterByDateRange(filterData, globalStartDate, globalEndDate) {
    // 结果对象，key为字符串，value为数组
    const result = {};

    // 遍历第一层
    for (const [outerKey, outerValue] of Object.entries(filterData)) {
        // 临时数组存储符合条件的键和其匹配日期数量
        const matchedItems = [];

        // 遍历第二层
        for (const [innerKey, innerValue] of Object.entries(outerValue)) {
            // 计算在日期范围内的日期数量
            const matchedDates = innerValue.dates.filter(date =>
                date >= globalStartDate && date <= globalEndDate
            );

            // 如果有符合条件的日期，将该键加入临时数组
            if (matchedDates.length > 0) {
                matchedItems.push({
                    key: innerKey,
                    count: matchedDates.length
                });
            }
        }

        // 按照匹配日期数量降序排序
        matchedItems.sort((a, b) => b.count - a.count);

        // 提取排序后的键
        result[outerKey] = matchedItems.map(item => {
            return {
                "key": item.key,
                "count": item.count,
                "zh-cn": filterData[outerKey][item.key]['zh-cn']
            };
        });
    }

    globalfilterLabels = result;
    generateFilterOptions();
    sent_filter_data();
}
function sent_filter_data() {
    $.ajax({
        url: '/filter_papers',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            filterLists: window.filterLists,  // 原有的 filterLists
            startDate: globalStartDate,                // 新增变量1
            endDate: globalEndDate,                 // 新增变量2
            startScore:globalStartScore,
            endScore:globalEndScore,
            dataPath: dataPath
        }, (key, value) => {
            if (value instanceof Set) {
                return Array.from(value);     // 保持 Set 转换为数组的逻辑
            }
            return value;
        }),
        success: function (response) {
            paperDataGlobal = response
            changePage(1);
        if (globalStartScore==-1 )
        {

            let all_scores = [];
            Object.values(paperDataGlobal).forEach(arr => {
            if (arr.score) 
                all_scores.push(arr.score);
        });

        allScores = [...new Set(all_scores)].sort();
        minScore=allScores[0];
        maxScore=allScores[allScores.length-1];
        $("#slider-score").slider({
            range: true,
            min:minScore,
            max:maxScore,
            step: 1, // 一天的毫秒数
            values: [minScore, maxScore],
            slide: function (event, ui) {
                const startScore = ui.values[0];
                const endScore = ui.values[1];
                $("#score-range").text(
                    startScore + "-" + endScore
                );
                globalStartScore = startScore;
                globalEndScore = endScore;
                sent_filter_data()
            }
        });
        $("#score-range").text(
            minScore + "-" + maxScore
        );
    }
        globalStartScore = minScore;
        globalEndScore = maxScore;
            console.log('数据发送成功:', response);
        },
        error: function (error) {
            console.error('数据发送失败:', error);
        }
    });
}

function renderKeywords(safeId, keywords, endIndex = INIT_KEYWORD_COUNT) {
    const container = $('#' + safeId);
    container.empty();

    // 创建筛选列表存储选中的关键词
    if (!window.filterLists) {
        window.filterLists = new Set();
    }

    const visibleKeywords = keywords.slice(0, endIndex);
    visibleKeywords.forEach(keyword => {
        const badge = $('<span>')
            .addClass(`badge rounded-pill me-1 mb-1 _keyword `)
            .text(keyword['zh-cn'] + `(${keyword['count']})`)
            .click(function () {
                const filterList = window.filterLists;
                if (filterList.has(keyword)) {
                    // 移除关键词
                    filterList.delete(keyword);
                    $(this).removeClass('text-bg-primary').addClass('text-bg-primary-outline');
                } else {
                    // 添加关键词
                    if (filterList.size === 0) {
                        $(`._keyword`).removeClass('text-bg-primary').addClass('text-bg-primary-outline');
                    }
                    filterList.add(keyword);
                    $(this).removeClass('text-bg-primary-outline').addClass('text-bg-primary');
                }

                // 如果筛选列表为空，将所有关键词设置为选中状态
                if (filterList.size === 0) {
                    $(`._keyword`).removeClass('text-bg-primary-outline').addClass('text-bg-primary');
                }
                sent_filter_data();
            });
        if (window.filterLists.has(keyword) || window.filterLists.size === 0) {
            badge.addClass('text-bg-primary');
        } else {
            badge.addClass('text-bg-primary-outline');
        }
        container.append(badge);
    });
}


function toggleMore(safeId, keywords) {
    const container = $('#' + safeId);
    const moreBtn = $('<span class="more-btn ms-2">').text('展开').attr('id', safeId + 'More');
    const lessBtn = $('<span class="more-btn ms-2">').text('收起').attr('id', safeId + 'Less');// 初始状态
    container.after(lessBtn);
    container.after(moreBtn);
    lessBtn.hide();
    let currentCount = $('.' + safeId + '_keyword').length;

    // 展开更多按钮点击事件
    moreBtn.click(function () {
        const nextCount = Math.min(currentCount + EXPAND_KEYWORD_COUNT, keywords.length);
        renderKeywords(safeId, keywords, nextCount);
        currentCount = nextCount;

        // 如果显示数量超过10个，显示收起按钮
        if (currentCount > INIT_KEYWORD_COUNT) {
            lessBtn.show();
        }

        // 如果所有关键词都已显示，隐藏展开按钮
        if (currentCount >= keywords.length) {
            moreBtn.hide();
        }
    });

    // 收起按钮点击事件
    lessBtn.click(function () {
        const nextCount = Math.max(currentCount - EXPAND_KEYWORD_COUNT, INIT_KEYWORD_COUNT);
        currentCount = nextCount;
        renderKeywords(safeId, keywords, nextCount); // 重置为前10个
        if (currentCount <= INIT_KEYWORD_COUNT) {
            lessBtn.hide();
        }
        if (currentCount < keywords.length) {
            moreBtn.show();
        }
    });
}

function generateFilterOptions() {
    const form = $('#filterForm');
    form.empty();

    Object.keys(globalfilterLabels).forEach(level => {
        const safeId = level.replace(/\s+/g, '_');

        // 创建每个筛选项的容器
        const filterDiv = $('<div class="mb-3">').append(`<p>${level}</p>`);
        const levelDiv = $('<div>').attr('id', safeId);
        filterDiv.append(levelDiv);

        // 将筛选项添加到表单
        form.append(filterDiv);
        // 渲染关键词和绑定展开逻辑
        renderKeywords(safeId, globalfilterLabels[level]);
        toggleMore(safeId, globalfilterLabels[level]);
    });
}


// 页面加载时生成筛选项
// 修改表单提交处理
$(document).ready(function () {

    $('#filterForm').submit(function (e) {
        e.preventDefault();

        const selectedFilters = {};
        Object.keys(filterData).forEach(level => {
            const safeId = level.replace(/\s+/g, '_');
            selectedFilters[level] = [];
            $('#' + safeId).find('.badge.selected').each(function () {
                selectedFilters[level].push($(this).text());
            });
        });

        // AJAX 调用部分保持不变
        console.log('Selected filters:', selectedFilters);
    });
});